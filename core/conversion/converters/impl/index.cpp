#include <ATen/ATen.h>
#include <vector>
#include "NvInfer.h"
#include "c10/util/intrusive_ptr.h"
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
auto index_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::index.Tensor(Tensor self, Tensor?[] indices) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto input_dims = in->getDimensions();
                    auto ind = args[1].IValue()->toListRef();
                    TRTORCH_CHECK(
           (input_dims.nbDims>=ind.size()),
           "Dimension out of range (expected to be in range of [" << 0 << ", " << input_dims.nbDims
                                                                  << "], but got " << ind.size() << ")");
                    
                    // run index_select for each dimension
                    for(int32_t axis = 0; axis<ind.size(); axis++){
                        
                        auto indices = ind[axis].toTensor();
                        LOG_DEBUG("Dtype: " << indices.dtype());
                        
                        // when used with boolean mask tensor
                        if(indices.dtype()==torch::kBool){
                            indices = torch::nonzero(indices).squeeze();
                        }
                        
                        // index to access needs to be an at::Tensor
                        auto weights = Weights(ctx, indices.to(torch::kI32));
                        
                        // IConstantLayer to convert indices from Weights to ITensor
                        auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                        TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                        auto const_out = const_layer->getOutput(0);
                        
                        // IGatherLayer takes in input tensor, the indices, and the axis
                        // of input tensor to take indices from
                        auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                        TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                        auto gather_out = gather_layer->getOutput(0);
                        
                        // IShuffleLayer removes redundant dimensions
//                         auto shuffle_layer = ctx->net->addShuffle(*gather_out);
//                         TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
//                         shuffle_layer->setReshapeDimensions(util::squeezeDims(gather_out->getDimensions(), axis));
//                         shuffle_layer->setName(util::node_info(n).c_str());
//                         in = shuffle_layer->getOutput(0);
                        
                        in = gather_out;
                    }  

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }});
    
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch