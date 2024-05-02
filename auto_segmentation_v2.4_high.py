#!/usr/bin/env python3
#folder: /script_1
##v 2.1- new alignment templates 
##v 2.2- post_NUC code is seperated from alignment
##v 2.3- defrag code is added
##v 2.4- Brain masking model is updated

### add error messages; 
### terminate processing if an I file does not exist
 
    
import numpy as np
import math
from numpy import zeros
import nibabel as nib
import matplotlib.pyplot as plt
import os
import csv
import sys
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
# subparsers = parser.add_subparsers(dest="subcommand")

parser.add_argument('--input_fol',
    nargs='?',			##// use + ?
    required=True,
    help='relative path to the data directory ($file variable)')

parser.add_argument('--masking', '--mask',
    dest='masks', 
	action='store_true',
    help='creates masks of the raw scans and moves the mask files into a folder')

parser.add_argument('--remask', 
    dest='remask', 
	action='store_true',
    help='creates brain folder and extracts brain region using manually corrected masks')

parser.add_argument('--NUC', '--nuc', 
    dest='NUC', 
	action='store_true',
    help='performs non uniformity corrrection')

parser.add_argument('--QA', '--qa',
    dest='QA',
	action='store_true',
    help='creates quality assessment .csv')

parser.add_argument('--recon', '--reconstruction',
    dest='recon',
	action='store_true',
    help='performs high resolution(0.5mm)reconstructions using Nesvor')

parser.add_argument('--alignment', '--align',
    dest='align',
	action='store_true',
    help='aligns the reconstructed images')

parser.add_argument('--segment', '--segmentation',
    dest='auto_seg',
	action='store_true',
    help='automatically segments the reconstructed images')

parser.add_argument('--all', 
    dest='all',
	action='store_true',
    help='does all steps from masking')

parser.add_argument('--from_remask', 
    dest='remask__',
	action='store_true',
    help='extracts corrected brain region and does following steps')

parser.add_argument('--from_NUC', 
    dest='nuc__',
	action='store_true',
    help='NUC to auto segmentation')

parser.add_argument('--from_QA', 
    dest='qa__',
	action='store_true',
    help='QA to segmentation')

parser.add_argument('--from_recon', 
    dest='recon__',
	action='store_true',
    help='recon to segmentation')

parser.add_argument('--from_alignment', 
    dest='align__',
	action='store_true',
    help='alignment to segmentation')

# parser.add_argument('--reprocess', 
#   dest='rm',
# 	action='store_true',
#   help='removes files and folders except masks/ and raw/')

def verify():
    img_list = np.asarray(sorted(glob.glob(input_fol+'/verify/*_brain.nii*')))
    def auto_crop_image(input_name, output_name, reserve):
        nim = nib.load(input_name)
        image = nim.get_data()
        if np.mean(image) == 0:
            print(input_name,'\t Passed')
            return 0
        # else:
        #     print(input_name, '\t Worked')
        image = np.pad(image, [(50,50),(50,50),(16,16)], 'constant')
        X, Y, Z = image.shape[:3]

        # Detect the bounding box of the foreground
        idx = np.nonzero(image > 0)
        x1, x2 = idx[0].min() - reserve[0,0], idx[0].max() + reserve[0,1] + 1
        y1, y2 = idx[1].min() - reserve[1,0], idx[1].max() + reserve[1,1] + 1
        z1, z2 = idx[2].min() - reserve[2,0], idx[2].max() + reserve[2,1] + 1
        # print('Bounding box')
        # print(input_name+'\t'+str([x2-x1, y2-y1, z2-z1]))
        # return [x2-x1, y2-y1, z2-z1]
        # print('  bottom-left corner = ({},{},{})'.format(x1, y1, z1))
        # print('  top-right corner = ({},{},{})'.format(x2, y2, z2))

        # Crop the image
        image = image[x1:x2, y1:y2, z1:z2]

        # Update the affine matrix
        affine = nim.affine
        affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
        nim2 = nib.Nifti1Image(image, affine)
        nib.save(nim2, output_name)
        return image

    for i in range(len(img_list)):
        f,axarr = plt.subplots(1,6)#,figsize=(len(in_img_list),9))
        f.patch.set_facecolor('k')
        #imsize = np.zeros([len(in_img_list),3])
        img = auto_crop_image(img_list[i], img_list[i].replace('.nii.gz','').replace('.nii','')+'_crop.nii.gz', np.array([[0,0],[0,0],[0,0]]))
        if isinstance(img,(list, tuple, np.ndarray)) == False:
            continue
        
        img = nib.load(img_list[i].replace('.nii.gz','').replace('.nii','')+'_crop.nii.gz').get_data()
        hdr = nib.load(img_list[i].replace('.nii.gz','').replace('.nii','')+'_crop.nii.gz').header
        axarr[0].imshow(np.rot90(img[:,:,np.int_(img.shape[-1]*0.3)]),cmap='gray')
        axarr[0].axis('off')
        axarr[0].set_title(str(img_list[i]),size=5,color='white')

        axarr[1].imshow(np.rot90(img[:,:,np.int_(img.shape[-1]*0.4)]),cmap='gray')
        axarr[1].axis('off')

        axarr[2].imshow(np.rot90(img[:,:,np.int_(img.shape[-1]*0.5)]),cmap='gray')
        axarr[2].axis('off')

        axarr[3].imshow(np.rot90(img[:,:,np.int_(img.shape[-1]*0.6)]),cmap='gray')
        axarr[3].axis('off')

        axarr[4].imshow(np.rot90(img[:,np.int_(img.shape[-2]*0.5),:]),cmap='gray',aspect=str(hdr['pixdim'][3]/hdr['pixdim'][2]))
        axarr[4].axis('off')

        axarr[5].imshow(np.rot90(img[np.int_(img.shape[0]*0.5),:,:]),cmap='gray',aspect=str(hdr['pixdim'][3]/hdr['pixdim'][1]))
        axarr[5].axis('off')

        f.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(img_list[i].replace('.nii.gz','').replace('.nii','')+'_crop_verify.png', facecolor=f.get_facecolor(), pad_inches=0, dpi=300)
        plt.close()
    return 0

    def _interslice_inorm(img_array):
        for i in range(img_array.shape[-1]):
            if i==0 :
                n_mean = np.mean(img_array[:,:,i+1][img_array[:,:,i+1]>0])
            elif i==img_array.shape[-1]-1:
                n_mean = np.mean(img_array[:,:,i-1][img_array[:,:,i-1]>0])
            else:
                n_mean = np.mean(img_array[:,:,[i-1,i+1]][img_array[:,:,[i-1,i+1]]>0])
            loc = np.where(img_array[:,:,i])
            img_array[:,:,i][loc]=img_array[:,:,i][loc]-np.mean(img_array[:,:,i][loc])+n_mean
        return img_array

    def _3dN4(img_array):
        import SimpleITK as sitk
        sitk_img = sitk.GetImageFromArray(img_array)
        maskImage = sitk.GetImageFromArray((img_array>0).astype(int))
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
        output = corrector.Execute( sitk_img, maskImage )
        img_array = sitk.GetArrayFromImage(output)
        return img_array

    def _2dN4(img_array):
        import SimpleITK as sitk
        result = np.zeros(np.shape(img_array))
        for i in range(result.shape[-1]):
            sitk_img = sitk.GetImageFromArray(img_array[:,:,i])
            maskImage = sitk.GetImageFromArray((img_array[:,:,i]>0).astype(int))
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
            maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
            output = corrector.Execute( sitk_img, maskImage )
            result[:,:,i] = sitk.GetArrayFromImage(output)
        return result

#create masks in raw folder, then move mask.nii into /masks and moves masked region into /brain
def masks():
    
    # os.system('/neuro/labs/grantlab/research/HyukJin_MRI/code/brain_mask2 --target-dir ./'+input_fol+'/raw')		##<< previous masking code// >> using sofia's apptainer ?replace with individual_brain_mask.py?
    
    os.system('singularity run --no-home -B ./'+input_fol+'/raw:/data /neuro/labs/grantlab/research/MRI_processing/sungmin.you/MRI_SIF/brain_mask.sif /data;')
    os.system('mv '+input_fol+'/raw/*mask.nii '+input_fol+'/masks;')
    
    # Binalize mask
    mask_list = glob.glob(input_fol+'/masks/*mask.nii')
    for file in mask_list:W
        if not file.endswith('~'):
            #Apply fslmaths for the binarization of the file
            #print(f"Binarizing mask: {file}")
            os.system('fslmaths ' + file + ' -thr ' +'0.001 ' + '-bin ' + file)

            #We remove the original file
            os.remove(file)

            #Decompress the new binarized file
            gzip_file = file + '.gz'
            #print(f"Unizpping mask: {gzip_file}")
            if os.path.exists(gzip_file):
                os.system('gunzip ' + gzip_file)
        
    img_list= np.asarray(sorted(glob.glob(input_fol+'/masks/*mask.nii')))
    for i in range(len(img_list)):
        vol = nib.load(img_list[i])
        vol_data = vol.get_data()
        if np.max(vol_data)>0.01:
            os.system('mri_mask '+img_list[i].replace('masks/','raw/').replace('_mask.nii','.nii')+' '+img_list[i]+'  '+img_list[i].replace('masks/','brain/').replace('_mask.nii','_brain.nii'))

    ##verify images
    os.system('cp -r ./'+input_fol+'/brain/ ./'+input_fol+'/verify/;')
    verify()

### REMASKING option// create new brain folder after [manual] mask correction 
def remask():
    if os.path.exists(input_fol+'/masks') and os.path.exists(input_fol+'/raw'):
        img_list= np.asarray(sorted(glob.glob(input_fol+'/masks/*mask.nii')))
        for i in range(len(img_list)):
            vol = nib.load(img_list[i])
            vol_data = vol.get_data()
            if np.max(vol_data)>0.01:
                os.system('mri_mask '+img_list[i].replace('masks/','raw/').replace('_mask.nii','.nii')+' '+img_list[i]+'  '+img_list[i].replace('masks/','brain/').replace('_mask.nii','_brain.nii'))
    
    ##verify images
    os.system('cp -r ./'+input_fol+'/brain/ ./'+input_fol+'/verify/;')
    verify()

#NUC
def nuc():
    img_list = np.asarray(sorted(glob.glob(input_fol+'/brain/*.nii')))
    for i in range(len(img_list)):
        os.system('~/arch/Linux64/packages/ANTs/current/bin/N4BiasFieldCorrection -d 3 -o '+img_list[i].replace('/brain/','/nuc/')+' -i '+img_list[i])

#Quality assessment 
def qa():
    os.system('singularity exec --no-home docker://fnndsc/pl-fetal-brain-assessment:1.3.0 fetal_brain_assessment ./'+input_fol+'/nuc/ ./'+input_fol+'/;')
    ##add python3/neuro/labs/grantlab/research/HyukJin_MRI/code/QA_resample.py ${file}/; for exceeding dimensions (similar to Andrea's)

#reconstruction - Nesvor
def recon():
    threshold = 0.4
    Best_list = []
    with open(input_fol+'/quality_assessment.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            score = float(row["quality"]) #Reading QC score and cast it to float
            if score >= threshold: 
                Best_list.append(row)

    Best_list_sorted = sorted(Best_list, key=lambda row: row['quality'], reverse=True)
  
    #print([row['filename'] for row in Best_list_sorted])
    #print(str([row['filename'] for row in Best_list_sorted]).replace("[","").replace("]","").replace("'","").replace(",",""))
    
    # print('CUDA_VISIBLE_DEVICES=0 apptainer exec --nv "/neuro/labs/grantlab/research/Apptainer/nesvor:v0.6.0rc2.sif" \
    #     nesvor reconstruct --input-stacks '+ str([row['filename'] for row in Best_list_sorted]).replace("[","").replace("]","").replace("'","").replace(",","")+ \
    #     ' --output-volume '+input_fol+'/recon/recon.nii \
    #     --output-resolution 0.5 --svort-version v2')

    os.system ('CUDA_VISIBLE_DEVICES=0 apptainer exec --nv "/neuro/labs/grantlab/research/Apptainer/nesvor:v0.6.0rc2.sif" \
        nesvor reconstruct --input-stacks '+ str([row['filename'] for row in Best_list_sorted]).replace("[","").replace("]","").replace("'","").replace(",","")+ \
        ' --output-volume '+input_fol+'/recon/recon.nii \
        --output-resolution 0.5 --svort-version v2')

#alignment
def align():
    os.system('python3 /neuro/users/mri.team/Codes/alignment_highres.py '+ input_fol+'/recon')

# NUC after alignment
def Post_NUC():
    os.system('cp '+input_fol+'/recon/alignment_temp/recon_to31.* '+input_fol+'/recon')

    os.system('~/arch/Linux64/packages/ANTs/current/bin/N4BiasFieldCorrection -d 3 -o '+input_fol+'/recon/recon_to31_nuc.nii -i '+input_fol+'/recon/recon_to31.nii;')

    os.system('convert_xfm -omat '+input_fol+'/recon/recon_to31_inv.xfm -inverse '+input_fol+'/recon/recon_to31.xfm;')
    os.system('echo `avscale '+input_fol+'/recon/recon_to31_inv.xfm | grep Scales` > '+input_fol+'/temp/temp.txt;')
    scales=open(input_fol+'/temp/temp.txt', encoding='utf-8')
    scales=scales.read()
    os.system('param2xfm -clobber -scales '+scales[16:-1]+' '+input_fol+'/recon/recon_native.xfm;')


#auto segmentation
def auto_seg():
    os.system('cp '+input_fol+'/recon/recon_to31_nuc.nii '+input_fol+'/segmentation')
    os.system('singularity run --no-home -B '+input_fol+'/segmentation/:/data --nv /neuro/labs/grantlab/research/MRI_processing/sungmin.you/MRI_SIF/fetal_cp_seg_0.5.sif recon_to31_nuc.nii . 0;')

#Defrag segmentation
def defrag_seg(MRI_input, MRI_output, opt_ratio=0.33):
    os.system('python3 /neuro/users/mri.team/Codes/Defrag_segmentation_HJ.py {0} {1} {2}'.format(MRI_input, MRI_output, opt_ratio))


def main():
    args = parser.parse_args()
    print(args)

    global input_fol
    input_fol = args.input_fol		## input_fol should be: ./$file

    #create subfolders inside main data directory 
    foldr=['raw', 'masks', 'brain', 'nuc', 'temp', 'recon', 'segmentation']
    for items in foldr:
	    os.makedirs(input_fol+'/'+items, exist_ok=True)
           
    #move raw scans into 'raw' folder from data dir
    dest=(input_fol+'/raw')
    for nii_file in glob.glob(input_fol+'/*.nii'):
        shutil.move(nii_file, dest) 

    masking = args.masks
    remasking = args.remask
    NUC = args.NUC
    QA = args.QA
    reconstruction = args.recon
    alignment = args.align
    seg = args.auto_seg
    from_remask = args.remask__
    fromNUC = args.nuc__
    fromQA = args.qa__
    from_recon = args.recon__
    from_align = args.align__
    allSteps = args.all
    # rm = args.rm


    if masking == True:
        masks()
    if remasking == True: 
        remask()
    if NUC == True:
        nuc()
    if QA == True:
        qa()
    if reconstruction == True:
        recon()
    if alignment == True:
        align()
    if seg == True:
        auto_seg()

    if from_remask == True:
        remask()
        nuc()
        qa()
        recon()
        align()
        Post_NUC()
        auto_seg()
        defrag_seg(input_fol+'/segmentation/recon_to31_nuc_deep_agg.nii.gz', input_fol+'/segmentation/recon_to31_nuc_deep_agg_defrag.nii.gz')

    if fromNUC == True:
        nuc()
        qa()
        recon()
        align()
        Post_NUC()
        auto_seg()
        defrag_seg(input_fol+'/segmentation/recon_to31_nuc_deep_agg.nii.gz', input_fol+'/segmentation/recon_to31_nuc_deep_agg_defrag.nii.gz')

    if fromQA == True:
        qa()
        recon()
        align()
        Post_NUC()
        auto_seg()
        defrag_seg(input_fol+'/segmentation/recon_to31_nuc_deep_agg.nii.gz', input_fol+'/segmentation/recon_to31_nuc_deep_agg_defrag.nii.gz')

    if from_recon == True:
        recon()
        align()
        Post_NUC()
        auto_seg()
        defrag_seg(MRI_input, MRI_output)

    if from_align == True:
        align()
        Post_NUC()
        auto_seg()
        defrag_seg(input_fol+'/segmentation/recon_to31_nuc_deep_agg.nii.gz', input_fol+'/segmentation/recon_to31_nuc_deep_agg_defrag.nii.gz')
        
    if allSteps == True:
        masks()
        nuc()
        qa()
        recon()
        align()
        Post_NUC()
        auto_seg()
        defrag_seg(input_fol+'/segmentation/recon_to31_nuc_deep_agg.nii.gz', input_fol+'/segmentation/recon_to31_nuc_deep_agg_defrag.nii.gz')

    # if rm == True:
    #     if os.path.exists(input_fol+'/temp_recon_1'):
    #         if input("Delete temp_recon files? (y/n)") == "y":
    #             os.system('rm -r '+input_fol+'/temp_recon_?/')
    #         exit()
    #     if input("Delete all other files & folders except for masks/ and raw/? (y/n)") == "y":
    #         os.system('rm -r '+input_fol+'/Best_Images_crop/ '+input_fol+'/brain/ '+input_fol+'/nuc/ '+input_fol+'/quality_assessment.csv '+input_fol+'/recon/ '+input_fol+'/temp_recon_?/')
    #         exit()
    #         if input("are you sure? (y/n)") != "y":
    #             exit()

    
    ####check if the functions run without all flag:
    # if masking and NUC and QA and reconstruction and alignment and seg and resize == False: 
    #     masks()
    #     nuc()
    #     qa()
    #     recon()
    #     align()
    #     auto_seg()
    #     transform()

if __name__ == '__main__':
    main()

