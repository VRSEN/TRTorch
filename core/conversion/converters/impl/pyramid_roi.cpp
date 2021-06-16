#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto nms TRTORCH_UNUSED = RegisterNodeConversionPatterns()
    .pattern({
        "torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            nvinfer1::ITensor* input = args[0].ITensor();
            nvinfer1::ITensor* rois = args[1].ITensor();
            auto spatial_scale = args[2].unwrapToDouble();
            auto pooled_height = args[3].unwrapToInt();
            auto pooled_width = args[4].unwrapToInt();
            auto sampling_ratio = args[5].unwrapToInt();
            auto aligned = args[6].unwrapToBool();
            
            TRTORCH_CHECK(pooled_height==pooled_width, "Different pool width and height");
            
            const auto inputs_dims = util::toVec(input->getDimensions());
           
            // unsqueeze input (batch dimension=1)
//             auto shuffle_layer_input = ctx->net->addShuffle(*input);
//             TRTORCH_CHECK(shuffle_layer_input, "Unable to create shuffle layer from node: " << *n);
//             shuffle_layer_input->setReshapeDimensions(util::unsqueezeDims(input->getDimensions(), 0));
//             input = shuffle_layer_input->getOutput(0);
            
            // reshape roi from [rois, 5] to [N, rois, 4]
            // 1) index [:, 1:]
            at::Tensor indices = torch::tensor({1,2,3,4}).to(torch::kI32);
            auto weights = Weights(ctx, indices);
            auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
            TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
            auto const_out = const_layer->getOutput(0);
            
            auto gather_layer = ctx->net->addGather(*rois, *const_out, 1);
            TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
            rois = gather_layer->getOutput(0);
            
            // 2) unsqueeze(0)
            auto shuffle_layer_roi = ctx->net->addShuffle(*rois);
            TRTORCH_CHECK(shuffle_layer_roi, "Unable to create shuffle layer from node: " << *n);
            shuffle_layer_roi->setReshapeDimensions(util::unsqueezeDims(rois->getDimensions(), 0));
            rois = shuffle_layer_roi->getOutput(0);
            
            // 3) unsqueeze(-1)
            auto shuffle_layer_roi2 = ctx->net->addShuffle(*rois);
            TRTORCH_CHECK(shuffle_layer_roi2, "Unable to create shuffle layer from node: " << *n);
            shuffle_layer_roi2->setReshapeDimensions(util::unsqueezeDims(rois->getDimensions(), util::toVec(rois->getDimensions()).size()));
            rois = shuffle_layer_roi2->getOutput(0);
            
             
//             LOG_DEBUG("inputs num dims: "<< inputs.nbDims << ", dim 3 has size: "<<boxes_dims.d[2]);
            LOG_DEBUG("input dims: "<< input->getDimensions() << ", rois dims: "<<rois->getDimensions());
            
            // Initialize.
//             nvinfer1::plugin::NMSParameters params{};
//             params.shareLocation =  true;
            
            std::vector<nvinfer1::PluginField> f;
            f.emplace_back("crop_height", &pooled_height, nvinfer1::PluginFieldType::kINT32, 1);
            f.emplace_back("crop_width", &pooled_width, nvinfer1::PluginFieldType::kINT32, 1);
            
            nvinfer1::PluginFieldCollection fc;
            fc.nbFields = f.size();
            fc.fields = f.data();
            
            // Create plugin from registry
            auto creator = getPluginRegistry()->getPluginCreator("CropAndResize", "1");
            auto plugin = creator->createPlugin("CropAndResizePluginTRTorch", &fc);   
//             LOG_DEBUG("plugin type " << plugin->getPluginType());
            
            LOG_DEBUG("PLUGIN CREATED");
            std::vector<nvinfer1::ITensor*> in = {input, rois};
//             auto nms_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 2, *plugin); 
            auto roi_layer = ctx->net->addPluginV2(in.data(), 2, *plugin); 
            TRTORCH_CHECK(roi_layer, "Unable to create layer for torchvision::roi_align");
            
            roi_layer->setName(util::node_info(n).c_str());
            
            auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], roi_layer->getOutput(0));
            LOG_DEBUG("roi_align layer output tensor shape: " << layer_output->getDimensions());
            return true;
        }
    });
    
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch