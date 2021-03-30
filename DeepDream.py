from keras.applications import inception_v3
from keras.preprocessing import image
from keras import backend as k
import numpy as np 
import scipy

k.set_learning_phase(0)
model = inception_v3.InceptionV3(weights='imagenet',include_top=False)

layer_contributions = {
    'mixed5':0.5,
    'mixed6':0.5,
    'mixed7':0.5,
}

#model.summary()

layer_dict = dict([(layer.name,layer) for layer in model.layers])
loss = k.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = k.prod(k.cast(k.shape(activation),'float32'))
    loss += coeff * k.sum(k.square(activation[:,2:-2,:2:-2,:])) / scaling

dream = model.input
grads = k.gradients(loss,dream)[0]
grads /= k.maximum(k.mean(k.abs(grads)),1e-7)
outputs = [loss,grads]
fetch_loss_and_grads = k.function([dream],outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_value = outs[1]
    return loss_value,grad_value

def gradient_ascent(x,iterations,step,max_loss=None):
    for i in range(iterations):
        loss_value,grad_value = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('第{}次循环，损失值为{}。'.format(i,loss_value))
        x += step * grad_value
    return x

def resize_img(img,size):
    img = np.copy(img)
    factors = (1,float(size[0])/img.shape[1],float(size[1])/img.shape[2],1)
    return scipy.ndimage.zoom(img,factors,order=1)

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    if k.image_data_format() == 'channels_first':
        x = x.reshape((3,x.shape[2],x.shape[3]))
        x = x.transpose((1,2,0))
    else:
        x = x.reshape((x.shape[1],x.shape[2],3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x,0,255).astype('uint8')
    return x

def save_img(img,fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname,pil_img)

step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 30
max_loss = 8

base_image_path = r'E:\20170111163957781.jpg'
img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1,num_octave):
    shape = tuple([int(dim / (octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    original_img = np.copy(img)
    shrunk_original_img = resize_img(img,successive_shapes[0])

    for shape in successive_shapes:
        print(shape)
        img = resize_img(img,shape)
        img = gradient_ascent(img,iterations,step,max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img,shape)
        same_size_original = resize_img(original_img,shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img,shape)
    save_img(img,fname='final_dream.png')
