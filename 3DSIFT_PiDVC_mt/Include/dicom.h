/* -----------------------------------------------------------------------------
* dicom.h
* -----------------------------------------------------------------------------
* Copyright (c) 2015-2016 Blaine Rister et al., see LICENSE for details.
* -----------------------------------------------------------------------------
* Internal header file for the DCMTK wrapper.
* -----------------------------------------------------------------------------
*/
 #include "SIFT3D.h"
 #ifndef _DICOM_H
#define _DICOM_H
 /* Length of UID buffers */
#define SIFT3D_UID_LEN 1024 
 /* Dicom file extension */
const char ext_dcm[] = "dcm";
extern const char ext_dcm[]; // dicom.h
const char ext_analyze[] = "img";;
const char ext_gz[] = "gz";
const char ext_nii[] = "nii";
const char ext_dir[] = "";
 /* Internal struct to hold limited Dicom metadata */
typedef struct _Dcm_meta {
	const char *patient_name; // Patient name
	const char *patient_id; // Patient ID
	const char *series_descrip; // Series description
	char study_uid[SIFT3D_UID_LEN]; // Study Instance UID
	char series_uid[SIFT3D_UID_LEN]; // Series UID
	char instance_uid[SIFT3D_UID_LEN]; // SOP Instance UID
	int instance_num; // Instance number
} Dcm_meta;
 /* Supported image file formats in original program, here we only use directory */
typedef enum _im_format {
	ANALYZE, /* Analyze */
	DICOM, /* DICOM */
	DIRECTORY, /* Directory */
	NIFTI, /* NIFTI-1 */
	UNKNOWN, /* Not one of the known extensions */
	FILE_ERROR /* Error occurred in determining the format */
} im_format;
 im_format im_get_format(const char *path);
 int read_dcm_dir_cpp(const char *path, Image * im);
 int im_resize(Image *const im);
 #endif