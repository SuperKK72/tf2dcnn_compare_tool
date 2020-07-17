import numpy as np
import sys
np.set_printoptions(suppress=True)

def read_tf_result(tf_result_filename):
    tf_result_list = []
    tf_result = open(tf_result_filename)
    for line in tf_result:
        tf_result_list.append(float(line))
    return tf_result_list

def tf_onnx_mse_compute(tf_result_list, onnx_result_list):
    assert len(tf_result_list) == len(onnx_result_list), 'Error: unequal list length.'
    total_num = len(tf_result_list)
    se = 0.
    for num_index in range(total_num):
        se += (tf_result_list[num_index] - onnx_result_list[num_index])**2
    mse = se / total_num
    return mse
if __name__ == '__main__':
    tf_result_path = './tf_inference_result/' + sys.argv[1].strip() + '_tf_inference_result' + '.txt'
    tf_result_list = read_tf_result(tf_result_path)
    tf_max_value = max(tf_result_list)
    tf_max_value_index = np.argmax(tf_result_list)
    tf_result_length = len(tf_result_list)
    # print(tf_result_list)

    dcnn_result_path = './dcnn_inference_result/' + sys.argv[1].strip() + '_dcnn_inference_result' + '.txt'
    dcnn_result_list = read_tf_result(dcnn_result_path)
    dcnn_max_value = max(dcnn_result_list)
    dcnn_max_value_index = np.argmax(dcnn_result_list)
    dcnn_result_length = len(dcnn_result_list)
    # print(tf_result_list)

    if(tf_result_length < dcnn_result_length):
        dcnn_result_list = dcnn_result_list[:tf_result_length]

    mse = tf_onnx_mse_compute(tf_result_list, dcnn_result_list)

    error_list = []
    for i in range(tf_result_length):
        error_list.append(abs(tf_result_list[i] - dcnn_result_list[i]))
    tf2dcnn_max_error = max(error_list)
    tf2dcnn_max_error_index = np.argmax(error_list)
    tf_max_error_value = tf_result_list[tf2dcnn_max_error_index]
    dcnn_max_error_value = dcnn_result_list[tf2dcnn_max_error_index]
    #tf2dcnn_error_scale = tf2dcnn_max_error / abs(tf_max_error_value)


    tf2dcnn_compare_result = ""
    if (tf_max_value_index == dcnn_max_value_index and tf2dcnn_max_error < 0.001):
        tf2dcnn_compare_result = "pass"
    else:
        tf2dcnn_compare_result = "error"
    


    # print(mse)

    result_saved_path = './tf2dcnn_compare_result/' + sys.argv[1].strip() + '_tf2dcnn_compare_result' + '.txt'
    with open(result_saved_path,'w') as file:
        file.write("tf2dcnn_compare_result: " + str(tf2dcnn_compare_result) + '\n')
        file.write("tf_max_value: " + str(tf_max_value) + '\n')
        file.write("tf_max_value_index: " + str(tf_max_value_index) + '\n')
        file.write("dcnn_max_value: " + str(dcnn_max_value) + '\n')
        file.write("dcnn_max_value_index: " + str(dcnn_max_value_index) + '\n')
        file.write("mse: " + str('{:f}'.format(mse)) + '\n')
        file.write("tf2dcnn_max_error: " + str('{:f}'.format(tf2dcnn_max_error)) + '\n')
        file.write("tf2dcnn_max_error_index: " + str(tf2dcnn_max_error_index) + '\n')
        file.write("tf_max_error_value: " + str(tf_max_error_value) + '\n')
        file.write("dcnn_max_error_value: " + str(dcnn_max_error_value) + '\n')
        # file.write('\n')
    file.close()
