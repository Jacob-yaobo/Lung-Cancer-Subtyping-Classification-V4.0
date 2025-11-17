import os
import numpy as np
import glob
import skimage.io as ski
import nibabel as nib
from nibabel import processing
import shutil
import pandas as pd
import dicom2nifti 
from datetime import datetime
import skimage
import skimage.transform as skitran
import logging
# import ants
import gc
import subprocess
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_erosion
# from seg_labels import total_label, total_mr, tissue_label
from multiprocessing import Pool, Manager
from scipy.ndimage import zoom
from joblib import parallel_backend
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

nnUnet_env_dir = '/home/song/nnUNet/nnUnet_global_env'

def main():
	data_dir = '/mnt/HDD_1/FDG/LungCancer_Subtyping/data/nifti/Neimeng_nifti_425'
	save_dir = '/mnt/HDD_1/FDG/LungCancer_Subtyping/data/seg/Neimeng_nifti_425'

	# segment_CT(data_dir, save_dir, seg_task='total')
	# segment_CT(data_dir, save_dir, seg_task='tissue_types')

	segment_lesion_PET(data_dir, save_dir)

	# segment_brain(data_dir, save_dir)

def segment_CT(data_dir, save_dir, seg_task):
	print('CT seg {}'.format(seg_task))
	df = pd.DataFrame()
	if seg_task == 'heartchambers_highres':
		save_name = 'heart'
	elif seg_task == 'tissue_types':
		save_name = 'tissue'
	elif seg_task == 'total':
		save_name = 'organ'

	for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
	# for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')][::-1]:
		print(pid)
		try:
			### Seg CT
			CT_file = os.path.join(os.path.join(data_dir,pid,'CT.nii.gz'))
			save_path = os.path.join(save_dir,pid)
			ct_seg_file = os.path.join(save_path,'CT_{}_seg.nii.gz'.format(save_name))
			if os.path.exists(CT_file):
				if not os.path.exists(ct_seg_file):
					os.makedirs(save_path, exist_ok=True)
					totalsegmentator(
									input=CT_file,
									output=ct_seg_file,
									task=seg_task,
									device = "gpu",
									ml=True,
									nr_thr_resamp=4,
									nr_thr_saving=4,
					)
			### Resample PET seg
			PET_file = os.path.join(os.path.join(data_dir,pid,'PET.nii.gz'))
			pet_seg_file = os.path.join(save_path,'PET_{}_seg.nii.gz'.format(save_name))
			if os.path.exists(PET_file) and os.path.exists(ct_seg_file):
				if not os.path.exists(pet_seg_file):
					PET_nii = nib.load(PET_file)
					seg_nii = nib.load(ct_seg_file)
					PET_lesion_seg = resmaple_seg(seg_nii, PET_nii)
					nib.save(nib.Nifti1Image(PET_lesion_seg, PET_nii.affine), pet_seg_file)
		except Exception as e:
			print('\t\tERROR: {}'.format(pid))
			continue

def segment_lesion_PET(data_dir, save_dir):
	# Set environment variable for nnUNet results
	os.environ['nnUNet_results'] = os.path.join(nnUnet_env_dir, 'nnUNet_results')
	os.environ['nnUNet_raw'] = os.path.join(nnUnet_env_dir, 'nnUNet_raw')
	os.environ['nnUNet_preprocessed'] = os.path.join(nnUnet_env_dir, 'nnUNet_preprocessed')
	task_id = 11

	count = 0 
	for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
		image_path = os.path.join(data_dir,pid,'PET.nii.gz')
		if os.path.exists(image_path):
			save_path = os.path.join(save_dir,pid)
			save_file = os.path.join(save_path,'PET_lesion_seg.nii.gz')
			if not os.path.exists(save_file):
				os.makedirs(save_path, exist_ok=True)
				count += 1
				print(pid, count)
				# Define paths and parameters
				output_path = os.path.join(save_path,'tmp')
				os.makedirs(output_path, exist_ok=True)
				tmp_dir = os.path.join(nnUnet_env_dir, 'tmp_lesion')
				os.makedirs(tmp_dir, exist_ok=True)
				shutil.copyfile(image_path, os.path.join(tmp_dir, 'image_0000.nii.gz'))
				# Run inference
				try:
					infer_command = [
						'nnUNetv2_predict',
						'-i', tmp_dir,
						'-o', output_path,
						'-d', str(task_id),
						'-f', 'all',
						'-c', '3d_fullres',
						'--disable_tta'
					]
					subprocess.run(infer_command, check=True)
					shutil.move(os.path.join(output_path,'image.nii.gz'), save_file)
					shutil.rmtree(output_path)
				except:
					continue

def segment_brain(data_dir, seg_dir):
	for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
		print(pid)
		pet_path = os.path.join(data_dir,pid,'PET_Brain.nii.gz')
		save_file_path = os.path.join(seg_dir,pid,'PET_brain_seg.nii.gz')
		if os.path.exists(pet_path):
			if not os.path.exists(save_file_path):
				try:
					nnUnet_brain_seg(pet_path, save_file_path)
				except:
					continue

def nnUnet_brain_seg(image_path, save_file_path):
	task_id = 13
	# Set environment variable for nnUNet results
	os.environ['nnUNet_results'] = os.path.join(nnUnet_env_dir, 'nnUNet_results')
	os.environ['nnUNet_raw'] = os.path.join(nnUnet_env_dir, 'nnUNet_raw')
	os.environ['nnUNet_preprocessed'] = os.path.join(nnUnet_env_dir, 'nnUNet_preprocessed')

	output_path = os.path.join(os.path.dirname(save_file_path),'tmp')
	os.makedirs(output_path, exist_ok=True)
	tmp_dir = os.path.join(nnUnet_env_dir, 'tmp')
	os.makedirs(tmp_dir, exist_ok=True)
	shutil.copyfile(image_path, os.path.join(tmp_dir, 'image_0000.nii.gz'))
	# Run inference
	infer_command = [
		'nnUNetv2_predict',
		'-i', tmp_dir,
		'-o', output_path,
		'-d', str(task_id),
		'-f', 'all',
		'-c', '3d_fullres',
		'--disable_tta'
	]
	subprocess.run(infer_command, check=True)
	shutil.move(os.path.join(output_path,'image.nii.gz'), save_file_path)
	shutil.rmtree(output_path)


def resmaple_seg(seg_nii, target_nii):
	seg = np.array(seg_nii.dataobj)
	target_shape = (seg.shape[0] // (target_nii.header.get_zooms()[0]/seg_nii.header.get_zooms()[0]), 
					seg.shape[1] // (target_nii.header.get_zooms()[1]/seg_nii.header.get_zooms()[1]), 
					seg.shape[2] // (target_nii.header.get_zooms()[2]/seg_nii.header.get_zooms()[2]))

	seg = skitran.resize(seg, target_shape, mode='edge',
							anti_aliasing=False,
							anti_aliasing_sigma=None,
							preserve_range=True,
							order=0)
	seg = np.around(seg)
	seg = seg.astype('uint8')
	if seg.shape[0] > target_nii.shape[0]:
		cut_num = int((seg.shape[0] - target_nii.shape[0]) / 2)
		seg = seg[cut_num:-(seg.shape[0] - target_nii.shape[0] - cut_num), :, :]
	elif seg.shape[0] < target_nii.shape[0]:
		pad_num = int((target_nii.shape[0] - seg.shape[0]) / 2)
		seg = np.pad(seg, ((pad_num,(target_nii.shape[0] - seg.shape[0])-pad_num),(0,0),(0,0)),'constant', constant_values=(0, 0))
	if seg.shape[1] > target_nii.shape[1]:
		cut_num = int((seg.shape[1] - target_nii.shape[1]) / 2)
		seg = seg[:, cut_num:-(seg.shape[1] - target_nii.shape[1] - cut_num), :]
	elif seg.shape[1] < target_nii.shape[1]:
		pad_num = int((target_nii.shape[1] - seg.shape[1]) / 2)
		seg = np.pad(seg, ((0,0),(pad_num,(target_nii.shape[1] - seg.shape[1])-pad_num),(0,0)),'constant', constant_values=(0, 0))
	if seg.shape[2] > target_nii.shape[2]:
		cut_num = int((seg.shape[2] - target_nii.shape[2]) / 2)
		seg = seg[:,:,cut_num:-(seg.shape[2] - target_nii.shape[2] - cut_num)]
	elif seg.shape[2] < target_nii.shape[2]:
		pad_num = int((target_nii.shape[2] - seg.shape[2]) / 2)
		seg = np.pad(seg, ((0,0),(0,0),(pad_num,(target_nii.shape[2] - seg.shape[2])-pad_num)),'constant', constant_values=(0, 0))
	return seg

if __name__ == '__main__':
	main()

