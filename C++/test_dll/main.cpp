# include "tf_clf.h"

int main() {
	string model_path = "D:/yeyan/pycharm_project/dogcat/model/model.pt";
	string img_path = "D:/yeyan/pycharm_project/dogcat/data/test_set/test_set/cats/cat.4001.jpg";
	TFClf clf = TFClf(img_path, model_path);
	clf.model_pred();


}