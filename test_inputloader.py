import inputloader

iloader = inputloader.InputLoader(batch_num=2)

image_batch, gt_batch, _ = iloader.get_image_and_gt()

print("back to main code")
print(gt_batch)


print("end of code")