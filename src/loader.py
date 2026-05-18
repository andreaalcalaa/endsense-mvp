import nibabel as nib

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return img, data