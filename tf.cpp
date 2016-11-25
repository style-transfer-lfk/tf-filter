#include <stdio.h>
#include <stdint.h>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

static std::unique_ptr<tensorflow::Session> session_seurat;
static std::unique_ptr<tensorflow::Session> session_composition;
static const int TF_CHANNEL_SIZE = 3;

extern "C" void tf_init()
{
    tensorflow::SessionOptions options_seurat;
    session_seurat.reset(tensorflow::NewSession(options_seurat));

    tensorflow::GraphDef graph_def_seurat;
    FILE *fp = fopen("/home/ec2-user/graph/graph-seurat.pb", "r");
    int fd = fileno(fp);

    fseek(fp, 0L, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    google::protobuf::io::FileInputStream is_seurat(fd);
    google::protobuf::io::LimitingInputStream lis_seurat(&is_seurat, sz);
    lis_seurat.Skip(0);
    graph_def_seurat.ParseFromZeroCopyStream(&lis_seurat);
    is_seurat.Close();

    tensorflow::Status s = session_seurat->Create(graph_def_seurat);
    if (!s.ok()) {
        printf("tf initialize error: %s\n", s.error_message().c_str());
        return;
    }
    graph_def_seurat.Clear();
    fclose(fp);

    tensorflow::SessionOptions options_composition;
    session_composition.reset(tensorflow::NewSession(options_composition));

    tensorflow::GraphDef graph_def_composition;
    fp = fopen("/home/ec2-user/graph/graph-composition.pb", "r");
    fd = fileno(fp);

    fseek(fp, 0L, SEEK_END);
    sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    google::protobuf::io::FileInputStream is_composition(fd);
    google::protobuf::io::LimitingInputStream lis_composition(&is_composition, sz);
    lis_composition.Skip(0);
    graph_def_composition.ParseFromZeroCopyStream(&lis_composition);
    is_composition.Close();

    s = session_composition->Create(graph_def_composition);
    if (!s.ok()) {
        printf("tf initialize error: %s\n", s.error_message().c_str());
        return;
    }
}

extern "C" void tf_transfer(uint8_t *pixels, int width, int height, const char *model_name)
{
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, height, width, TF_CHANNEL_SIZE }));
    auto input_tensor_mapped = input_tensor.flat<float>();
    int pixel_size = width * height;
    for (int i = 0; i < pixel_size; i++) {
        input_tensor_mapped(i * 3 + 0) = pixels[i * 3 + 0];
        input_tensor_mapped(i * 3 + 1) = pixels[i * 3 + 1];
        input_tensor_mapped(i * 3 + 2) = pixels[i * 3 + 2];
    }

    std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors({{ "input", input_tensor }});
    std::vector<tensorflow::Tensor> output_tensors;
    std::vector<std::string> output_names({ "output" });
    tensorflow::Status run_status;
    if (model_name[0] == 's') {
        run_status = session_seurat->Run(input_tensors, output_names, {}, &output_tensors);
    } else {
        run_status = session_composition->Run(input_tensors, output_names, {}, &output_tensors);
    }
    if (!run_status.ok()) {
        printf("error %s!!\n", run_status.error_message().c_str());
        return;
    }

    tensorflow::Tensor &output_tensor = output_tensors[0];
    tensorflow::TTypes<unsigned char>::Flat output_flat = output_tensor.flat<unsigned char>();
    for (int i = 0; i < pixel_size; i++) {
        pixels[i * 3 + 0] = output_flat(i * 3 + 0);
        pixels[i * 3 + 1] = output_flat(i * 3 + 1);
        pixels[i * 3 + 2] = output_flat(i * 3 + 2);
    }
}
