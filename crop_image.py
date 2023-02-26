from PIL import Image


image = Image.open('data/test/NYC_LOS_V.png')
# image = Image.open('data/test/c24_Chittagong_ASC_20181_30m_2.png')


x_center = image.width // 2
y_center = image.height // 2
left = x_center - 50
top = y_center - 50
right = x_center + 50
bottom = y_center + 50
cropped_image = image.crop((left, top, right, bottom))


# cropped_image.save('data/test/c24_Chittagong_ASC_20181_30m_2_center100.png')
cropped_image.save('data/test/NYC_LOS_V_center100.png')