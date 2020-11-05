#include "pch.h"
#include "tf_clf.h"


Mat TFClf::preprocess_img(Mat img) {
    //ͼ�����resize����
    resize(img, img, cv::Size(resize_col, resize_row));

    //ȥ��ֵ
    img.convertTo(img, CV_32FC3);
    // �������е����ص�ÿ��ͨ������ȥ��Ӧ�ľ�ֵ��BGR����˳��
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            img.at<Vec3f>(row, col)[0] -= mean[0];
            img.at<Vec3f>(row, col)[1] -= mean[1];
            img.at<Vec3f>(row, col)[2] -= mean[2];
        }
    }
    return img;
}

void TFClf::mat_to_tensor(Mat img, Tensor* output_tensor) {
    //����һ��ָ��tensor�����ݵ�ָ��
    float* p = output_tensor->flat<float>().data();

    //����һ��Mat����tensor��ָ���,�ı����Mat��ֵ�����൱�ڸı�tensor��ֵ
    cv::Mat tempMat(resize_row, resize_col, CV_32FC3, p);
    img.convertTo(tempMat, CV_32FC3);
}

void TFClf::show_result_pic(Mat img, int output_class_id,double output_prob) {
    string rs_string;
    if (output_class_id == 0) {
        rs_string = "Class:Dog Proba:" + to_string(output_prob);
    }
    else {
        rs_string = "Class:Cat Proba:" + to_string(output_prob);
    }
    // ��label�����ͼƬ��
    cv::putText(img, rs_string, draw_point, FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
    cv::imshow("result", img);
    cv::waitKey(0);
}

void TFClf::model_pred() {
    /*--------------------------------����session------------------------------*/
    Session* session;
    Status status = NewSession(SessionOptions(), &session);//�����»ỰSession

    /*--------------------------------��pb�ļ��ж�ȡģ��--------------------------------*/
    GraphDef graphdef; //Graph Definition for current model

    Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //��pb�ļ��ж�ȡͼģ��;
    if (!status_load.ok()) {
        std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
        std::cout << status_load.ToString() << "\n";
    }
    Status status_create = session->Create(graphdef); //��ģ�͵���ỰSession��;
    if (!status_create.ok()) {
        std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
    }
    std::cout << "<----Successfully created session and load graph.------->" << endl;

    /*---------------------------------�������ͼƬ-------------------------------------*/
    std::cout << endl << "<------------loading test_image-------------->" << endl;
    Mat img = imread(image_path);


    if (img.empty())
    {
        std::cout << "can't open the image!!!!!!!" << endl;
    }
    // ͼ��Ԥ����
    Mat resize_img;
    resize_img = preprocess_img(img);
    //����һ��tensor��Ϊ��������Ľӿ�
    Tensor resized_tensor(DT_FLOAT, TensorShape({ 1,resize_row,resize_col,3 }));

    //��Opencv��Mat��ʽ��ͼƬ����tensor
    mat_to_tensor(resize_img, &resized_tensor);

    /*-----------------------------------��������в���-----------------------------------------*/
    std::cout << endl << "<-------------Running the model with test_image--------------->" << endl;
    //ǰ�����У�������һ����һ��tensor��vector
    vector<tensorflow::Tensor> outputs;
    string output_node = output_tensor_name;
    Status status_run = session->Run({ {input_tensor_name, resized_tensor} }, { output_node }, {}, &outputs);

    if (!status_run.ok()) {
        std::cout << "ERROR: RUN failed..." << std::endl;
        std::cout << status_run.ToString() << "\n";
    }
    //�����ֵ����ȡ����
    Tensor t = outputs[0];                   // Fetch the first tensor
    auto tmap = t.tensor<float, 2>();        // Tensor Shape: [batch_size, target_class_num]
    int output_dim = t.shape().dim_size(1);  // Get the target_class_num from 1st dimension
    // Argmax: Get Final Prediction Label and Probability
    int output_class_id = -1;
    double output_prob = 0.0;
    for (int j = 0; j < output_dim; j++)
    {
        std::cout << "Class " << j << " prob:" << tmap(0, j) << "," << std::endl;
        if (tmap(0, j) >= output_prob) {
            output_class_id = j;
            output_prob = tmap(0, j);
        }
    }
    // չʾͼƬ���
    show_result_pic(img, output_class_id, output_prob);

}
