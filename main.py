from create_net import create_training_net
import tensorflow as tf
from inputloader import InputLoader


# load the input and GT

# input_batch
# gt_batch


g1,notable_tensors, input_holders = create_training_net()

inputloader = InputLoader()

image_input, gt, _ = inputloader.get_image_and_gt()

with tf.Session(graph=g1) as sess:
    writer= tf.summary.FileWriter(logdir="./summary",graph=sess.graph)
    writer.flush()

    # wanted_outputs={
    #     'net_out': g1.get_tensor_by_name("net_out"),
    #     'coord_pred' : g1.get_operation_by_name("coord_pred")
    # }

    

    feed_dict = {
        input_holders['input_layer'] : image_input,
        input_holders['ground_truth'] : gt
    }
    
    sess.run(tf.global_variables_initializer())

    fetches=[notable_tensors['coord_pred'], notable_tensors['conf_pred'], 
        notable_tensors['pclass_pred'],
        notable_tensors['loss_coords']
        
        ]

    coord_pred, conf_pred, pclass_pred, loss_coords = sess.run(fetches,feed_dict=feed_dict)

    
    # print('coord_preds', coord_pred)
    print('loss_coords',loss_coords)



print("end of code")