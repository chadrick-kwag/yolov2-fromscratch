import inputloader


iloader = inputloader.InputLoader(batch_num=4)


steps = 1000

for step in range(steps):

    print("step={}".format(step))
    image_batch, gt_batch, epoch_end_signal , essence= iloader.get_image_and_gt()

    sample_image = image_batch[0]
    print("sample_image shape = ", sample_image.shape)
    if epoch_end_signal:
        print("EPOCH ENDED!")


