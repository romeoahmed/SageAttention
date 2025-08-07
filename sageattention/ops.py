import torch

try:
    from . import _qattn_sm80
    _qattn_sm80 = torch.ops.sageattention_qattn_sm80
    SM80_ENABLED = True
except Exception:
    _qattn_sm80 = None
    SM80_ENABLED = False

try:
    from . import _qattn_sm89
    _qattn_sm89 = torch.ops.sageattention_qattn_sm89
    SM89_ENABLED = True
except Exception:
    _qattn_sm89 = None
    SM89_ENABLED = False

try:
    from . import _qattn_sm90
    _qattn_sm90 = torch.ops.sageattention_qattn_sm90
    SM90_ENABLED = True
except Exception:
    _qattn_sm90 = None
    SM90_ENABLED = False

if SM80_ENABLED:
    def sm80_qk_fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        if tensor_layout == 0:
            num_qo_heads = query.size(2)
            qo_len = query.size(1)
        else:
            num_qo_heads = query.size(1)
            qo_len = query.size(2)

        if return_lse:
            lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device="cuda")
        else:
            lse = torch.empty((0))
        return lse

    torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f32_attn")(sm80_qk_fake_impl)
    torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn")(sm80_qk_fake_impl)
    torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_attn_inst_buf")(sm80_qk_fake_impl)

    @torch.library.register_fake("sageattention_qattn_sm80::qk_int8_sv_f16_accum_f16_fuse_v_mean_attn")
    def qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_mean: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return sm80_qk_fake_impl(
            query, key, value, output, query_scale, key_scale, tensor_layout,
            is_causal, qk_quant_gran, sm_scale, return_lse
        )

if SM89_ENABLED:
    def sm89_qk_fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        if tensor_layout == 0:
            num_qo_heads = query.size(2)
            qo_len = query.size(1)
        else:
            num_qo_heads = query.size(1)
            qo_len = query.size(2)

        if return_lse:
            lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device="cuda")
        else:
            lse = torch.empty((0))
        return lse

    torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn")(sm89_qk_fake_impl)
    torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf")(sm89_qk_fake_impl)
    torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf")(sm89_qk_fake_impl)

    @torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn")
    def qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        value_mean: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return sm89_qk_fake_impl(
            query, key, value, output, query_scale, key_scale, value_scale,
            tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
        )

if SM90_ENABLED:
    @torch.library.register_fake("sageattention_qattn_sm90::qk_int8_sv_f8_accum_f32_attn_inst_buf")
    def qk_int8_sv_f8_accum_f32_attn_inst_buf_fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        if tensor_layout == 0:
            num_qo_heads = query.size(2)
            qo_len = query.size(1)
        else:
            num_qo_heads = query.size(1)
            qo_len = query.size(2)

        if return_lse:
            lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device="cuda")
        else:
            lse = torch.empty((0))
        return lse

    @torch.library.register_fake("sageattention_qattn_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf")
    def qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return qk_int8_sv_f8_accum_f32_attn_inst_buf_fake_impl(
            query, key, value, output, query_scale, key_scale, tensor_layout,
            is_causal, qk_quant_gran, sm_scale, return_lse
        )
