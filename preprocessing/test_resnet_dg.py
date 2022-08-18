from resnet_dg import ResnetDataGenerator
import math
import cv2

def test_resnet_dg(true_csv_file , data_path):
    batch_size = 10
    num_instances = 10
    resnet_dg = ResnetDataGenerator(true_csv_file, data_path , batch_size, num_instances = num_instances , norm = False)
    for i in range(math.ceil(num_instances / batch_size)):
        images , labels = resnet_dg.__getitem__(i)
        for img,lbl in zip(images , labels):
            lbl = "car" if lbl[0] else "background"
            print("label : " , lbl)
            cv2.imshow("img" , img[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    true_csv_file = "/home/yogeesh/yogeesh/datasets/car/data/train_solution_bounding_boxes.csv"
    data_path = "/home/yogeesh/yogeesh/datasets/car/data/training_images"
    test_resnet_dg(true_csv_file , data_path)
