import os
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import time
import sys
import tensorflow as tf



#默认为0：输出所有log信息
#设置为1：进一步屏蔽INFO信息
#设置为2：进一步屏蔽WARNING信息
#设置为3：进一步屏蔽ERROR信息




# print("Parameter num: ", len(sys.argv))
# print("Parameters:    ", sys.argv)
# print("Script name:   ", sys.argv[0])
# for arg_index in range(1, len(sys.argv)):
#     print("Parameter name: ", sys.argv[arg_index])

network_config_file_path = './tf_inference_config/' + sys.argv[1]
network_config_file = open(network_config_file_path)
network_config_parameters = []
for network_config_parameter in network_config_file:
    if len(network_config_parameter.strip()) != 0:
        # print(network_config_parameter)
        network_config_parameter = network_config_parameter[network_config_parameter.find('=')+1:].strip()
        if network_config_parameter[0] != '#':
            network_config_parameters.append(network_config_parameter)
network_config_file.close()

net_name    = network_config_parameters[0]
net_path    = network_config_parameters[1]
input_name  = network_config_parameters[2]
output_name = network_config_parameters[3]
img_path    = network_config_parameters[4]


img_n       = int(network_config_parameters[5])
img_c       = int(network_config_parameters[6])
img_h       = int(network_config_parameters[7])
img_w       = int(network_config_parameters[8])
std         = float(network_config_parameters[9])
val_B       = float(network_config_parameters[10])
val_G       = float(network_config_parameters[11])
val_R       = float(network_config_parameters[12])
val_D       = float(network_config_parameters[13])

rlt_path    = network_config_parameters[14]



def run_tf():

    with tf.Graph().as_default():

        graph_def = tf.GraphDef()
        with open(net_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input = sess.graph.get_tensor_by_name(input_name)
            output = sess.graph.get_tensor_by_name(output_name)

            image=cv2.imread(img_path, -1)
            image=cv2.resize(image, (img_w, img_h))
            # print(image)
            X=np.array(image).astype(np.float32)
            X[:, :, 0] = (X[:, :, 0] - val_B) * std
            X[:, :, 1] = (X[:, :, 1] - val_G) * std
            X[:, :, 2] = (X[:, :, 2] - val_R) * std
            if img_c == 4:
                X[:, :, 3] = (X[:, :, 3] - val_D) * std
            X=X.reshape((img_n, img_h, img_w, img_c))

            start = time.time()
            results = sess.run(output, feed_dict={input: X})
            end = time.time()
            # print(type(results))
            # print(results.shape)
            # print(len(results.shape))
            # print(results.shape[1])
            #print(results)
            #print(len(results.flatten()))

            print("net name: ", net_name)
            print("input size: ", X.shape)
            print("input name: ", input_name)
            print("output name: ", output_name)
            print("output shape: ", results.shape)
            print("time of runtime: ", end-start, "s")
            print("len of result: ", len(results.flatten()))
            print("max of result: ", max(results.flatten()))



            output_shape = results.shape
            output_dim = len(results.shape)
            f = open(rlt_path, 'w')
            if (output_dim == 4):
                #os.system("pause")
                h = output_shape[1]
                w = output_shape[2]
                c = output_shape[3]
                max_index = np.argmax(results.flatten())
                rest = max_index % c
                max_index = max_index / c + rest
                print("index of max result: ", int(max_index))
                results = results.flatten()
                totol_size = results.size
                map_size = h * w
                print("----> start to save!")
                for i in range(c):
                    for j in range(map_size):
                        curr_point_index = j * c + i
                        f.write(str(results[curr_point_index]))
                        f.write('\n')
                f.close()
            else:
                print("index of max result: ", np.argmax(results.flatten()))
                print("----> start to save!")
                for value in results[0].flatten():
                    f.write(str(value))
                    f.write('\n')
                f.close()
            print("----> results have been saved!")

if __name__ == "__main__":
    run_tf()