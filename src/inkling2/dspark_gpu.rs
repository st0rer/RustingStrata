//! Inkling 2 DSpark GPU speculative drafter (100% VRAM offload).
//!
//! Executes DSpark non-causal block transformer layers and the chained Markov RNN head
//! directly on the GPU without intermediate PCIe host round-trips.

use crate::backend::{Backend, Buf, GpuTensor};
use crate::gguf::Gguf;
use crate::quant::GgmlType;
use anyhow::{Context, Result, ensure};

use super::dspark::DsparkDrafter;

pub struct GpuDsparkLayer {
    pub swa: bool,
    pub dflare_norm: Buf,
    pub attn_k_ctx: GpuTensor,
    pub attn_v_ctx: GpuTensor,
    pub attn_k_ctx_norm: Buf,
    pub attn_norm: Buf,
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    pub attn_q_norm: Buf,
    pub attn_k_norm: Buf,
    pub ffn_norm: Buf,
    pub ffn_gate: GpuTensor,
    pub ffn_up: GpuTensor,
    pub ffn_down: GpuTensor,
    pub k_cache: Buf,
    pub v_cache: Buf,
    pub ring: usize,
}

pub struct GpuDspark {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rank: usize,
    pub block_size: usize,
    pub n_swa: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub softcap: f32,
    pub target_layers: [usize; 4],
    pub max_ctx: usize,
    pub kv_len: usize,

    pub fuse: Buf,
    pub dspark_norm: Buf,
    pub mask_embd: Buf,
    pub output_norm: Buf,

    pub markov_w1: Buf,
    pub joint_proj: GpuTensor,
    pub joint_proj_b: Buf,
    pub hidden_out: GpuTensor,
    pub hidden_out_b: Buf,

    pub layers: Vec<GpuDsparkLayer>,
    pub ctx_last: Buf,

    // Scratch buffers for draft execution:
    pub scr_inp: Buf,
    pub scr_h: Buf,
    pub scr_q: Buf,
    pub scr_k: Buf,
    pub scr_v: Buf,
    pub scr_attn: Buf,
    pub scr_gu: Buf,
    pub scr_hidden: Buf,
    pub scr_joint_in: Buf,
    pub scr_joint: Buf,
    pub scr_state: Buf,
    pub scr_latent: Buf,
    pub scr_corr: Buf,
    pub scr_out_tok: Buf,
    pub scr_anchor_emb: Buf,

    // Scratch buffers for inject_subset:
    pub scr_inj_ctx: Buf,
    pub scr_inj_k: Buf,
    pub scr_inj_v: Buf,
    pub max_inj_chunk: usize,
}

fn upload_f32(be: &dyn Backend, g: &Gguf, name: &str) -> Result<Buf> {
    let t = g.tensors.get(name).with_context(|| format!("dspark_gpu: missing tensor {name}"))?;
    let data = g.bytes(t);
    let buf = be.alloc(data.len())?;
    be.htod(data, &buf, 0)?;
    Ok(buf)
}

fn upload_tensor(be: &dyn Backend, g: &Gguf, name: &str) -> Result<GpuTensor> {
    let t = g.tensors.get(name).with_context(|| format!("dspark_gpu: missing tensor {name}"))?;
    let rows = t.dims[1..].iter().product::<u64>() as usize;
    let cols = t.dims[0] as usize;
    let ty = GgmlType::from_u32(t.ty)?;
    let data = g.bytes(t);
    let scale = name.strip_suffix(".weight")
        .and_then(|stem| g.tensors.get(&format!("{stem}.scale")))
        .filter(|st| st.ty == 0)
        .and_then(|st| g.bytes(st).try_into().ok())
        .map_or(1.0, f32::from_le_bytes);
    be.upload(&crate::weights::Tensor { ty, cols, rows, data, scale })
}

impl GpuDspark {
    pub fn new(be: &dyn Backend, ds: &DsparkDrafter, max_ctx: usize) -> Result<Self> {
        let g = ds.gguf();
        let (n_layer, n_embd, n_head, n_head_kv) = (ds.n_layer, ds.n_embd, ds.n_head, ds.n_head_kv);
        let (head_dim, n_ff, rank, block_size) = (ds.head_dim, ds.n_ff, ds.rank, ds.block_size);
        let (n_swa, eps, rope_theta, softcap) = (ds.n_swa, ds.eps, ds.rope_theta, ds.softcap);
        let target_layers = ds.target_layers;
        let kw = n_head_kv * head_dim;
        let qd = n_head * head_dim;
        let bs = block_size;

        // 1. Precompute normalized fuse weights
        let raw_fuse = ds.fuse();
        ensure!(raw_fuse.len() == 4 * (n_layer + 1), "dspark_fuse length mismatch");
        let mut norm_fuse = vec![0.0f32; raw_fuse.len()];
        for r in 0..=n_layer {
            let mut row = raw_fuse[r * 4..(r + 1) * 4].to_vec();
            crate::ops::softmax(&mut row);
            norm_fuse[r * 4..(r + 1) * 4].copy_from_slice(&row);
        }
        let fuse = be.alloc(4 * norm_fuse.len())?;
        be.put(&norm_fuse, &fuse, 0)?;

        let dspark_norm = upload_f32(be, g, "dspark_norm.weight")?;
        let mask_embd = upload_f32(be, g, "mask_embd.weight")?;
        let output_norm = upload_f32(be, g, "output_norm.weight")?;

        // 2. Markov RNN weights
        let markov_t = g.tensors.get("markov_w1.weight").context("missing markov_w1.weight")?;
        let markov_data = g.bytes(markov_t);
        let markov_w1 = be.alloc(markov_data.len())?;
        be.htod(markov_data, &markov_w1, 0)?;

        let joint_proj = upload_tensor(be, g, "joint_proj.weight")?;
        let joint_proj_b = upload_f32(be, g, "joint_proj.bias")?;
        let hidden_out = upload_tensor(be, g, "dspark_hidden_out.weight")?;
        let hidden_out_b = upload_f32(be, g, "dspark_hidden_out.bias")?;

        // 3. Layers & KV cache
        let swa_pat = g.get("dflash.attention.sliding_window_pattern")?
            .as_arr()
            .context("sliding_window_pattern")?;
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let is_swa = match &swa_pat[il] {
                crate::gguf::Value::Bool(b) => *b,
                other => other.as_u64().unwrap_or(0) != 0,
            };
            let ring = if is_swa { n_swa } else { max_ctx.min(32768) };
            let k_cache = be.alloc(4 * ring * kw)?;
            let v_cache = be.alloc(4 * ring * kw)?;
            layers.push(GpuDsparkLayer {
                swa: is_swa,
                dflare_norm: upload_f32(be, g, &p("dflare_norm.weight"))?,
                attn_k_ctx: upload_tensor(be, g, &p("attn_k_ctx.weight"))?,
                attn_v_ctx: upload_tensor(be, g, &p("attn_v_ctx.weight"))?,
                attn_k_ctx_norm: upload_f32(be, g, &p("attn_k_ctx_norm.weight"))?,
                attn_norm: upload_f32(be, g, &p("attn_norm.weight"))?,
                wq: upload_tensor(be, g, &p("attn_q.weight"))?,
                wk: upload_tensor(be, g, &p("attn_k.weight"))?,
                wv: upload_tensor(be, g, &p("attn_v.weight"))?,
                wo: upload_tensor(be, g, &p("attn_output.weight"))?,
                attn_q_norm: upload_f32(be, g, &p("attn_q_norm.weight"))?,
                attn_k_norm: upload_f32(be, g, &p("attn_k_norm.weight"))?,
                ffn_norm: upload_f32(be, g, &p("ffn_norm.weight"))?,
                ffn_gate: upload_tensor(be, g, &p("ffn_gate.weight"))?,
                ffn_up: upload_tensor(be, g, &p("ffn_up.weight"))?,
                ffn_down: upload_tensor(be, g, &p("ffn_down.weight"))?,
                k_cache,
                v_cache,
                ring,
            });
        }

        let ctx_last = be.alloc(4 * n_embd)?;

        // 4. Scratch buffers
        let scr_inp = be.alloc(4 * bs * n_embd)?;
        let scr_h = be.alloc(4 * bs * n_embd)?;
        let scr_q = be.alloc(4 * bs * qd)?;
        let scr_k = be.alloc(4 * bs * kw)?;
        let scr_v = be.alloc(4 * bs * kw)?;
        let scr_attn = be.alloc(4 * bs * qd)?;
        let scr_gu = be.alloc(4 * bs * 2 * n_ff)?;
        let scr_hidden = be.alloc(4 * bs * n_embd)?;
        let scr_joint_in = be.alloc(4 * (2 * rank + 2 * n_embd))?;
        let scr_joint = be.alloc(4 * 3 * rank)?;
        let scr_state = be.alloc(4 * rank)?;
        let scr_latent = be.alloc(4 * rank)?;
        let scr_corr = be.alloc(4 * n_embd)?;
        let scr_out_tok = be.alloc(4 * (block_size + 1))?;
        let scr_anchor_emb = be.alloc(4 * n_embd)?;

        let max_inj_chunk = 64;
        let scr_inj_ctx = be.alloc(4 * max_inj_chunk * n_embd)?;
        let scr_inj_k = be.alloc(4 * max_inj_chunk * kw)?;
        let scr_inj_v = be.alloc(4 * max_inj_chunk * kw)?;

        Ok(Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            rank,
            block_size,
            n_swa,
            eps,
            rope_theta,
            softcap,
            target_layers,
            max_ctx,
            kv_len: 0,
            fuse,
            dspark_norm,
            mask_embd,
            output_norm,
            markov_w1,
            joint_proj,
            joint_proj_b,
            hidden_out,
            hidden_out_b,
            layers,
            ctx_last,
            scr_inp,
            scr_h,
            scr_q,
            scr_k,
            scr_v,
            scr_attn,
            scr_gu,
            scr_hidden,
            scr_joint_in,
            scr_joint,
            scr_state,
            scr_latent,
            scr_corr,
            scr_out_tok,
            scr_anchor_emb,
            scr_inj_ctx,
            scr_inj_k,
            scr_inj_v,
            max_inj_chunk,
        })
    }

    pub fn truncate(&mut self, len: usize) {
        if len < self.kv_len {
            self.kv_len = len;
        }
    }

    pub fn inject_subset(
        &mut self,
        be: &dyn Backend,
        pos0: usize,
        keep: usize,
        stride_nt: usize,
        dspark_buf_addr: u64,
    ) -> Result<()> {
        if keep == 0 { return Ok(()); }
        let n = self.n_embd;
        let kw = self.n_head_kv * self.head_dim;
        let eps = self.eps;
        let theta = self.rope_theta;

        let chunk_size = self.max_inj_chunk;
        for t0 in (0..keep).step_by(chunk_size) {
            let cur_keep = (keep - t0).min(chunk_size);
            let cur_pos0 = pos0 + t0;

            for il in 0..self.n_layer {
                let l = &self.layers[il];
                let fuse_w_addr = self.fuse.addr() + (il * 4 * 4) as u64;
                let feats_addr = dspark_buf_addr + (4 * t0 * n) as u64;
                be.dspark_mix_norm(
                    feats_addr,
                    fuse_w_addr,
                    l.dflare_norm.addr(),
                    self.scr_inj_ctx.addr(),
                    cur_keep,
                    stride_nt,
                    n,
                    eps,
                )?;

                be.gemvq2(&l.attn_k_ctx, &l.attn_v_ctx, self.scr_inj_ctx.addr(), n, 0, self.scr_inj_k.addr(), self.scr_inj_v.addr(), cur_keep)?;

                be.dspark_norm_rope(
                    self.scr_inj_k.addr(),
                    l.attn_k_ctx_norm.addr(),
                    self.scr_inj_k.addr(),
                    self.head_dim,
                    self.n_head_kv,
                    cur_pos0,
                    cur_keep,
                    eps,
                    theta,
                )?;

                be.dspark_kv_store(
                    self.scr_inj_k.addr(),
                    self.scr_inj_v.addr(),
                    l.k_cache.addr(),
                    l.v_cache.addr(),
                    cur_pos0,
                    cur_keep,
                    kw,
                    l.ring,
                )?;
            }

            // Committed context vector
            let fuse_w_addr = self.fuse.addr() + (self.n_layer * 4 * 4) as u64;
            let feats_addr = dspark_buf_addr + (4 * t0 * n) as u64;
            be.dspark_mix_norm(
                feats_addr,
                fuse_w_addr,
                self.dspark_norm.addr(),
                self.scr_inj_ctx.addr(),
                cur_keep,
                stride_nt,
                n,
                eps,
            )?;

            let last_idx = cur_keep - 1;
            be.dtod(
                &self.scr_inj_ctx,
                last_idx * 4 * n,
                &self.ctx_last,
                0,
                4 * n,
            )?;
        }

        self.kv_len = pos0 + keep;
        Ok(())
    }

    pub fn draft(
        &mut self,
        be: &dyn Backend,
        target_output: &GpuTensor,
        s_logits: u64,
        n_vocab: usize,
        _softcap: f32,
        anchor_emb: &[f32],
        anchor_tok: u32,
        pos0: usize,
        max_draft: usize,
    ) -> Result<Vec<u32>> {
        let count = max_draft.min(self.block_size.saturating_sub(1));
        if count == 0 || pos0 == 0 || pos0 > self.kv_len {
            return Ok(Vec::new());
        }
        let (bs, n, nh, nkv, hd, nf) = (self.block_size, self.n_embd, self.n_head, self.n_head_kv, self.head_dim, self.n_ff);
        let (qd, _kw) = (nh * hd, nkv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = self.eps;
        let theta = self.rope_theta;

        // 1. Initialize input block [anchor_emb, 7 * mask_embd]
        be.put(anchor_emb, &self.scr_anchor_emb, 0)?;
        be.dspark_init_inp(self.scr_anchor_emb.addr(), self.mask_embd.addr(), self.scr_inp.addr(), n, bs)?;

        // 2. Run the 3 non-causal DSpark transformer layers
        for il in 0..self.n_layer {
            let l = &self.layers[il];
            be.rms_norm(self.scr_inp.addr(), n, l.attn_norm.addr(), self.scr_h.addr(), n, n, bs, eps)?;

            be.gemvq(&l.wq, self.scr_h.addr(), n, 0, self.scr_q.addr(), qd, bs)?;
            be.gemvq2(&l.wk, &l.wv, self.scr_h.addr(), n, 0, self.scr_k.addr(), self.scr_v.addr(), bs)?;

            be.dspark_norm_rope(self.scr_q.addr(), l.attn_q_norm.addr(), self.scr_q.addr(), hd, nh, pos0, bs, eps, theta)?;
            be.dspark_norm_rope(self.scr_k.addr(), l.attn_k_norm.addr(), self.scr_k.addr(), hd, nkv, pos0, bs, eps, theta)?;

            let first = if l.swa { pos0.saturating_sub(self.n_swa) } else { pos0.saturating_sub(l.ring) };
            let n_ctx = pos0 - first;
            be.dspark_attn(
                self.scr_q.addr(),
                l.k_cache.addr(),
                l.v_cache.addr(),
                self.scr_k.addr(),
                self.scr_v.addr(),
                self.scr_attn.addr(),
                first,
                n_ctx,
                l.ring,
                bs,
                nh,
                nkv,
                hd,
                scale,
            )?;

            be.gemvq(&l.wo, self.scr_attn.addr(), qd, 0, self.scr_h.addr(), n, bs)?;
            be.add(self.scr_inp.addr(), self.scr_h.addr(), bs * n)?;

            be.rms_norm(self.scr_inp.addr(), n, l.ffn_norm.addr(), self.scr_h.addr(), n, n, bs, eps)?;
            be.gemvq2(&l.ffn_gate, &l.ffn_up, self.scr_h.addr(), n, 0, self.scr_gu.addr(), self.scr_gu.addr() + 4 * (bs * nf) as u64, bs)?;
            be.swiglu(self.scr_gu.addr(), self.scr_gu.addr() + 4 * (bs * nf) as u64, bs * nf)?;
            be.gemvq(&l.ffn_down, self.scr_gu.addr(), nf, 0, self.scr_h.addr(), n, bs)?;
            be.add(self.scr_inp.addr(), self.scr_h.addr(), bs * n)?;
        }

        // 3. Output RMS norm -> scr_hidden
        be.rms_norm(self.scr_inp.addr(), n, self.output_norm.addr(), self.scr_hidden.addr(), n, n, bs, eps)?;

        // 4. Chained Markov RNN Head (Captured in CUDA Graph, 100% Async on GPU, zero PCIe host round trips per step)
        be.memset(&self.scr_state)?;
        let anchor_bytes = anchor_tok.to_le_bytes();
        be.put(&anchor_bytes, &self.scr_out_tok, 0)?;

        let rank = self.rank;

        let markov_key = 0x4453_504B_0000_0000u64 | (count as u64);
        be.graph(markov_key, &mut || {
            for step in 0..count {
                let h_row_addr = self.scr_hidden.addr() + 4 * ((step + 1) * n) as u64;
                let prev_tok_ptr = self.scr_out_tok.addr() + 4 * step as u64;
                let next_tok_ptr = self.scr_out_tok.addr() + 4 * (step + 1) as u64;

                be.dspark_markov_prep(
                    prev_tok_ptr,
                    self.markov_w1.addr(),
                    self.scr_state.addr(),
                    h_row_addr,
                    self.ctx_last.addr(),
                    self.scr_joint_in.addr(),
                    n,
                    rank,
                )?;

                be.gemvq(&self.joint_proj, self.scr_joint_in.addr(), 5120, 0, self.scr_joint.addr(), 768, 1)?;

                be.dspark_markov_step(
                    self.scr_joint.addr(),
                    self.joint_proj_b.addr(),
                    self.scr_state.addr(),
                    self.scr_latent.addr(),
                    rank,
                )?;

                be.gemvq(&self.hidden_out, self.scr_latent.addr(), 256, 0, self.scr_corr.addr(), n, 1)?;
                be.dspark_hidden_corr(self.scr_corr.addr(), self.hidden_out_b.addr(), h_row_addr, n)?;

                be.gemvq(target_output, self.scr_corr.addr(), n, 0, s_logits, n_vocab, 1)?;
                // softcap is strictly monotonic, so argmax_rows yields the exact same token without running tanh over 201K floats
                be.argmax_rows(s_logits, n_vocab, 1, next_tok_ptr)?;
            }
            Ok(())
        })?;

        // Single contiguous DtoH copy of all generated draft tokens at the end
        let mut drafts_bytes = vec![0u8; count * 4];
        be.dtoh(&self.scr_out_tok, 4, &mut drafts_bytes)?;
        let drafts: Vec<u32> = drafts_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(drafts)
    }
}
