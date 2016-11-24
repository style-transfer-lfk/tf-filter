#include <stdio.h>
#include <stdint.h>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

static std::unique_ptr<tensorflow::Session> session;
static const int TF_DISPLAY_WIDTH = 256;
static const int TF_DISPLAY_HEIGHT = 256;
static const int TF_PIXEL_SIZE = TF_DISPLAY_WIDTH * TF_DISPLAY_HEIGHT;
static const int TF_CHANNEL_SIZE = 3;

extern "C" void tf_init(const char *model)
{
    tensorflow::SessionOptions options;
    session.reset(tensorflow::NewSession(options));

    tensorflow::GraphDef graph_def;
    FILE *fp = fopen(model, "r");
    int fd = fileno(fp);

    fseek(fp, 0L, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    printf("model file size: %ld\n", sz);
    google::protobuf::io::FileInputStream is(fd);
    google::protobuf::io::LimitingInputStream lis(&is, sz);
    lis.Skip(0);
    graph_def.ParseFromZeroCopyStream(&lis);
    is.Close();

    tensorflow::Status s = session->Create(graph_def);
    if (!s.ok()) {
        printf("tf initialize error: %s\n", s.error_message().c_str());
        return;
    }

    printf("complete session creation by %s\n", model);
    graph_def.Clear();
}

extern "C" void tf_transfer(uint8_t *pixels)
{
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, TF_DISPLAY_HEIGHT, TF_DISPLAY_WIDTH, TF_CHANNEL_SIZE }));
    auto input_tensor_mapped = input_tensor.flat<float>();
    for (int i = 0; i < TF_PIXEL_SIZE; i++) {
        input_tensor_mapped(i * 3 + 0) = pixels[i * 3 + 0];
        input_tensor_mapped(i * 3 + 1) = pixels[i * 3 + 1];
        input_tensor_mapped(i * 3 + 2) = pixels[i * 3 + 2];
    }

    std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors({{ "input", input_tensor }});
    std::vector<tensorflow::Tensor> output_tensors;
    std::vector<std::string> output_names({ "output" });
    tensorflow::Status run_status = session->Run(input_tensors, output_names, {}, &output_tensors);
    if (!run_status.ok()) {
        printf("error %s!!\n", run_status.error_message().c_str());
        return;
    }

    tensorflow::Tensor &output_tensor = output_tensors[0];
    tensorflow::TTypes<unsigned char>::Flat output_flat = output_tensor.flat<unsigned char>();
    for (int i = 0; i < TF_PIXEL_SIZE; i++) {
        pixels[i * 3 + 0] = output_flat(i * 3 + 0);
        pixels[i * 3 + 1] = output_flat(i * 3 + 1);
        pixels[i * 3 + 2] = output_flat(i * 3 + 2);
    }
}
