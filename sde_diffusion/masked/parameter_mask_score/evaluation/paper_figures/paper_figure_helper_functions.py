from append_directories import *


def load_diffusion_images(model_name, image_name, file_name):

    eval_folder = append_directory(2)
    diffusion_images = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/diffusion/" + file_name + ".npy"))
    return diffusion_images

def load_mask(model_name, image_name):

    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/" + "mask.npy"))
    return mask

def load_reference_image(model_name, image_name):

    eval_folder = append_directory(2)
    ref_image = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy"))
    return ref_image

def load_observations(model_name, image_name, mask, n):

    eval_folder = append_directory(2)
    ref_image = np.load(eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy")
    observations = ref_image[(mask).astype(int) == 1]
    return observations

def produce_figure_name(model_name, image_name, fig_name):

    eval_folder = append_directory(2)
    figname = (eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name +
               "/paper_figures/" + fig_name)
    return figname