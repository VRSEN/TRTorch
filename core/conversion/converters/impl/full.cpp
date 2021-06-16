#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto cat_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto self = args[0].ITensor();
              auto shape = util::toVec(self->getDimensions());
              auto tensor_type = util::toATenDType(self->getType());
              auto options = torch::TensorOptions().dtype(tensor_type);
                
              auto fill_value = args[1].unwrapToScalar();
              auto t = at::full({shape}, fill_value, {options});
              auto t_weights = Weights(ctx, t);
              auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
              const_layer->setName(util::node_info(n).c_str());
              auto const_out = ctx->AssociateValueAndTensor(n->outputs()[0], const_layer->getOutput(0));

              return true;
            }});
    
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch