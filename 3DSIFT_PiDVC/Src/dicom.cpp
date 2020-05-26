#include "dicom.h"
#include "util.h"
#include "MemManager.h"
 #include "dcmtk/config/osconfig.h"    /* make sure OS specific configuration is included first */
 #define INCLUDE_CSTDIO
#define INCLUDE_CSTRING
#include "dcmtk/ofstd/ofstdinc.h"
 #ifdef HAVE_GUSI_H
#include <GUSI.h>
#endif
 #include "dcmtk/dcmdata/dctk.h"          /* for various dcmdata headers */
#include "dcmtk/dcmdata/cmdlnarg.h"      /* for prepareCmdLineArgs */
#include "dcmtk/dcmdata/dcuid.h"         /* for dcmtk version name */
 #include "dcmtk/ofstd/ofconapp.h"        /* for OFConsoleApplication */
#include "dcmtk/ofstd/ofcmdln.h"         /* for OFCommandLine */
 #include "dcmtk/oflog/oflog.h"           /* for OFLogger */
 #include "dcmtk/dcmimgle/dcmimage.h"     /* for DicomImage */
#include "dcmtk/dcmimgle/diutils.h"     /* for DIPixel */
#include "dcmtk/dcmimage/diregist.h"     /* include to support color images */
#include "dcmtk/dcmdata/dcrledrg.h"      /* for DcmRLEDecoderRegistration */
 #include "dcmtk/dcmjpeg/djdecode.h"      /* for dcmjpeg decoders */
#include "dcmtk/dcmjpeg/dipijpeg.h"      /* for dcmimage JPEG plugin */
 #include "dcmtk/dcmjpeg/djencode.h" /* for JPEG encoding */
#include "dcmtk/dcmjpeg/djrplol.h"  /* for DJ_RPLossless */
 #include "dcmtk/dcmseg/segment.h" /* Dicom segmentations */
 //#ifdef WITH_ZLIB
//#include <zlib.h>          /* for zlibVersion() */
//#endif
 /*---------------------------------------------------------*/
 /* Other includes */
#include <algorithm>
#include <memory>
#include <vector>
#include <cmath>
#include <cfloat>
#include <stdint.h>
#include <dirent.h>
 //Original macros for definition
#define SIFT3D_SUCCESS 0
#define SIFT3D_FAILURE -1
#define SIFT3D_ERR(...) fprintf(stderr, __VA_ARGS__)
 /* File separator character */
#define SIFT3D_FILE_SEP '\\'
 // Loop through an image in x, z, y order. Delmit with SIFT3D_IM_LOOP_END
#define SIFT3D_IM_LOOP_START(im, x, y, z) \
	for (z = 0; (z) < (im)->nz; (z)++) {	\
	for ((y) = 0; (y) < (im)->ny; (y)++) {	\
	for ((x) = 0; (x) < (im)->nx; (x)++) {
 // Get the index of an [x,y,z] pair in an image 
#define SIFT3D_IM_GET_IDX(im, x, y, z) ((size_t) (x) * (im)->xs + \
        (size_t) (y) * (im)->ys + (size_t) (z) * (im)->zs)
 // Get the value of voxel [x,y,z] in an image 
#define SIFT3D_IM_GET_VOX(im, x, y, z) ((im)->data[ \
        SIFT3D_IM_GET_IDX((im), (x), (y), (z))])
 // Delimit an SIFT3D_IM_LOOP_START or SIFT3D_IM_LOOP_LIMITED_START
#define SIFT3D_IM_LOOP_END }}}
 /* Helper class to store DICOM data. */
class Dicom {
private:
	std::string filename; // DICOM file name
	std::string classUID;
	std::string seriesUID; // Series UID 
	std::string instanceUID; // Instance UID
	double z; // z position in the series
	double ux, uy, uz; // Voxel spacing in real-world coordinates
	int nx, ny, nz, nc; // Image dimensions
	bool valid; // Data validity 
 public:
 	/* Data is initially invalid */
	Dicom() : valid(false) {};
 	~Dicom() {};
 	/* Load a file */
	Dicom(const char *filename);
 	/* Get the x-dimension */
	int getNx(void) const {
		return nx;
	}
 	/* Get the y-dimension */
	int getNy(void) const {
		return ny;
	}
 	/* Get the z-dimension */
	int getNz(void) const {
		return nz;
	}
 	/* Get the number of channels */
	int getNc(void) const {
		return nc;
	}
 	/* Get the x-spacing */
	double getUx(void) const {
		return ux;
	}
 	/* Get the y-spacing */
	double getUy(void) const {
		return uy;
	}
 	/* Get the z-spacing */
	double getUz(void) const {
		return uz;
	}
 	/* Check whether or not the data is valid */
	bool isValid(void) const {
		return valid;
	}
 	/* Get the file name */
	const char *name(void) const {
		return filename.c_str();
	}
 	/* Sort by z position */
	bool operator < (const Dicom &dicom) const {
		return z < dicom.z;
	}
 	/* Check if another DICOM file is from the same series */
	bool eqSeries(const Dicom &dicom) const {
		return !seriesUID.compare(dicom.seriesUID);
	}
 	/* Check for a matching SOPInstanceUID */
	bool eqInstance(const char *uid) const {
		return !instanceUID.compare(uid);
	}
 	/* Check if this is a Dicom Segmentation Object */
	bool isDSO(void) const {
		return !classUID.compare(UID_SegmentationStorage);
	}
};
 static int read_dcm_img(const Dicom &dicom, Image *const im);
 static int read_dcm_dir_meta(const char *path, std::vector<Dicom> &dicoms);
 static int dcm_resize_im(const std::vector<Dicom> &dicoms, Image *const im);
 static int copy_slice(const Image *const slice, const int off_z,
	Image *const volume);
 static int load_file(const char *path, DcmFileFormat &fileFormat);
 /* Read a DICOM file with DCMTK. */
static int load_file(const char *path, DcmFileFormat &fileFormat) {
 	// Load the image as a DcmFileFormat 
	OFCondition status = fileFormat.loadFile(path);
	if (status.bad()) {
		SIFT3D_ERR("load_file: failed to read DICOM file %s (%s)\n",
			path, status.text());
		return SIFT3D_FAILURE;
	}
 	return SIFT3D_SUCCESS;
}
 /* Load the data from a DICOM file */
Dicom::Dicom(const char *path) : filename(path), valid(false) {
 	// Read the file
	DcmFileFormat fileFormat;
	if (load_file(path, fileFormat))
		return;
	DcmDataset *data = fileFormat.getDataset();
 	// Get the SOPClass UID
	const char *classUIDStr;
	OFCondition status;
	/*status = data->findAndGetString(DCM_SOPClassUID,
		classUIDStr);
	if (status.bad() || classUIDStr == NULL) {
		SIFT3D_ERR("Dicom.Dicom: failed to get SOPClassUID "
			"from file %s (%s)\n", path, status.text());
		return;
	}
	classUID = std::string(classUIDStr);*/
 	// Get the series UID 
	const char *seriesUIDStr;
	status = data->findAndGetString(DCM_SeriesInstanceUID, seriesUIDStr);
	if (status.bad() || seriesUIDStr == NULL) {
		SIFT3D_ERR("Dicom.Dicom: failed to get SeriesInstanceUID "
			"from file %s (%s)\n", path, status.text());
		return;
	}
	seriesUID = std::string(seriesUIDStr);
 	// Get the instance UID 
	/*const char *instanceUIDStr;
	status = data->findAndGetString(DCM_SOPInstanceUID, instanceUIDStr);
	if (status.bad() || instanceUIDStr == NULL) {
		SIFT3D_ERR("Dicom.Dicom: failed to get SOPInstanceUID "
			"from file %s (%s)\n", path, status.text());
		return;
	}
	instanceUID = std::string(instanceUIDStr);*/
 	// Get the z coordinate
	if (isDSO()) {
		// DSOs don't always have z coordinates
		z = -1;
 		// DSOs don't always have units
		ux = uy = -1.0;
	}
	else {
#if 0
		// Read the patient position
		const char *patientPosStr;
		status = data->findAndGetString(DCM_PatientPosition,
			patientPosStr);
		if (status.bad() || patientPosStr == NULL) {
			SIFT3D_ERR("Dicom.Dicom: failed to get "
				"PatientPosition from file %s (%s)\n", path,
				status.text());
			return;
		}
 		// Interpret the patient position to give the sign of the z axis
		double zSign;
		switch (patientPosStr[0]) {
		case 'H':
			zSign = -1.0;
			break;
		case 'F':
			zSign = 1.0;
			break;
		default:
			SIFT3D_ERR("Dicom.Dicom: unrecognized patient "
				"position: %d\n", patientPosStr);
			return;
		}
#else
		//TODO: Is this needed?
		const double zSign = 1.0;
#endif
 		// Read the image position patient vector
		const char *imPosPatientStr;
		status = data->findAndGetString(DCM_ImagePositionPatient,
			imPosPatientStr);
		if (status.bad() || imPosPatientStr == NULL) {
			SIFT3D_ERR("Dicom.Dicom: failed to get "
				"ImagePositionPatient from file %s (%s)\n", path,
				status.text());
			return;
		}
 		// Parse the image position patient vector to get z
		double imPosZ;
		if (sscanf(imPosPatientStr, "%*f\\%*f\\%lf", &imPosZ) != 1) {
			SIFT3D_ERR("Dicom.Dicom: failed to parse "
				"ImagePositionPatient tag %s from file %s\n",
				imPosPatientStr, path);
			return;
		}
 		// Compute the z-location of the upper-left corner, in 
		// feet-first coordinates
		z = zSign * imPosZ;
 		// Read the pixel spacing
		const char *pixelSpacingStr;
		status = data->findAndGetString(DCM_PixelSpacing,
			pixelSpacingStr);
		if (status.bad()) {
			SIFT3D_ERR("Dicom.Dicom: failed to get pixel spacing "
				"from file %s (%s)\n", path, status.text());
			return;
		}
		if (sscanf(pixelSpacingStr, "%lf\\%lf", &ux, &uy) != 2) {
			SIFT3D_ERR("Dicom.Dicom: unable to parse pixel "
				"spacing from file %s \n", path);
			return;
		}
		if (ux <= 0.0 || uy <= 0.0) {
			SIFT3D_ERR("Dicom.Dicom: file %s has invalid pixel "
				"spacing [%f, %f]\n", path, ux, uy);
			return;
		}
 		// Read the slice thickness 
		Float64 sliceThickness;
		status = data->findAndGetFloat64(DCM_SliceThickness,
			sliceThickness);
		if (!status.good()) {
			SIFT3D_ERR("Dicom.Dicom: failed to get slice "
				"thickness from file %s (%s)\n", path,
				status.text());
			return;
		}
 		// Convert to double 
		uz = sliceThickness;
		if (uz <= 0.0) {
			SIFT3D_ERR("Dicom.Dicom: file %s has invalid slice "
				"thickness: %f \n", path, uz);
			return;
		}
	}
 	// Load the DicomImage object
	DicomImage dicomImage(path);
	if (dicomImage.getStatus() != EIS_Normal) {
		SIFT3D_ERR("Dicom.Dicom: failed to open image %s (%s)\n",
			path,
			DicomImage::getString(dicomImage.getStatus()));
		return;
	}
 	// Check for color images
	if (!dicomImage.isMonochrome()) {
		SIFT3D_ERR("Dicom.Dicom: reading of color DICOM images is "
			"not supported at this time \n");
		return;
	}
	nc = 1;
 	// Read the dimensions
	nx = dicomImage.getWidth();
	ny = dicomImage.getHeight();
	nz = dicomImage.getFrameCount();
	if (nx < 1 || ny < 1 || nz < 1) {
		SIFT3D_ERR("Dicom.Dicom: invalid dimensions for file %s "
			"(%d, %d, %d)\n", path, nx, ny, nz);
		return;
	}
 	// Set the window 
	dicomImage.setMinMaxWindow();
 	valid = true;
}
 /* File separator in string form */
const std::string sepStr(1, SIFT3D_FILE_SEP);
 ///* Dicom parameteres */
const unsigned int dcm_bit_width = 8; // Bits per pixel
 									  /* DICOM metadata defaults */
const char *default_patient_name = "DefaultSIFT3DPatient";
const char *default_series_descrip = "Series generated by SIFT3D";
const char *default_patient_id = "DefaultSIFT3DPatientID";
const char default_instance_num = 1;
 /* Separate the file name component from its path */
static const char *get_file_name(const char *path) {
 	const char *name;
 	// Get the last file separator
	name = strrchr(path, SIFT3D_FILE_SEP);
	return name == NULL ? path : name;
}
 ///* Get the extension of a file name */
static const char *get_file_ext(const char *name)
{
 	const char *dot;
 	// Get the file name component
	name = get_file_name(name);
 	// Get the last dot
	dot = strrchr(name, '.');
 	return dot == NULL || dot == name ? "" : dot + 1;
}
 int read_dcm_dir_cpp(const char *path, Image * im)
{
	int i, off_z;
 	// Read the DICOM metadata
	std::vector<Dicom> dicoms;
	if (read_dcm_dir_meta(path, dicoms))
		return SIFT3D_FAILURE;
 	// Initialize the image volume
	if (dcm_resize_im(dicoms, im))
		return SIFT3D_FAILURE;
 	// Allocate a temporary image for the slices
	Image slice;
 	init_im(&slice);
 	// Read the image data
	off_z = 0;
	const int num_files = dicoms.size();
	for (i = 0; i < num_files; i++) {
 		// Read and copy the slice data
		if (read_dcm_img(dicoms[i], &slice) ||
			copy_slice(&slice, off_z, im)) {
			im_free(&slice);
			return SIFT3D_FAILURE;
		}
		off_z += slice.nz;
	}
	//assert(off_z == im->nz);
	im_free(&slice);
 	return SIFT3D_SUCCESS;
}
 /* Helper function to read the metadata from a directory of DICOM files */
int read_dcm_dir_meta(const char *path, std::vector<Dicom> &dicoms) {
 	struct stat st;
	DIR *dir;
	struct dirent *ent;
 	// Verify that the directory exists
	if (stat(path, &st)) {
		SIFT3D_ERR("read_dcm_dir_cpp: cannot find file %s \n", path);
		return SIFT3D_FAILURE;
	}
	else if (!S_ISDIR(st.st_mode)) {
		SIFT3D_ERR("read_dcm_dir_cpp: file %s is not a directory \n",
			path);
		return SIFT3D_FAILURE;
	}
 	// Open the directory
	if ((dir = opendir(path)) == NULL) {
		SIFT3D_ERR("read_dcm_dir_cpp: unexpected error opening "
			"directory %s \n", path);
		return SIFT3D_FAILURE;
	}
 	// Get all of the .dcm files in the directory, ignoring DSOs
	dicoms.clear();
	while ((ent = readdir(dir)) != NULL) {
 		// Form the full file path
		std::string fullfile(std::string(path) + sepStr + ent->d_name);
 		// Check if it is a DICOM file 
		if (im_get_format(fullfile.c_str()) != DICOM)
			continue;
 		// Read the file
		Dicom dicom(fullfile.c_str());
		if (!dicom.isValid()) {
			closedir(dir);
			return SIFT3D_FAILURE;
		}
 		// Ignore DSOs
		if (dicom.isDSO())
			continue;
 		// Add the file to the list
		dicoms.push_back(dicom);
	}
 	// Release the directory
	closedir(dir);
 	// Verify that dicom files were found
	if (dicoms.size() == 0) {
		SIFT3D_ERR("read_dcm_dir_cpp: no DICOM files found in %s \n",
			path);
		return SIFT3D_FAILURE;
	}
 	// Sort the slices by z position
	std::sort(dicoms.begin(), dicoms.end());
 	return SIFT3D_SUCCESS;
}
 im_format im_get_format(const char *path)
{
	struct stat st;
	const char *ext;
 	// Check if the file exists and is a directory
	if (stat(path, &st) == 0) {
		if (S_ISDIR(st.st_mode))
			return DIRECTORY;
	}
 	// If not a directory, get the file extension
	ext = get_file_ext(path);
 	// Check the known types
	if (!strcmp(ext, ext_analyze) || !strcmp(ext, ext_gz) ||
		!strcmp(ext, ext_nii))
		return NIFTI;
 	if (!strcmp(ext, ext_dcm))
		return DICOM;
 	if (!strcmp(ext, ext_dir))
		return DIRECTORY;
 	// The type was not recognized
	return UNKNOWN;
}
 /* Resize an image to fit a DICOM series. */
static int dcm_resize_im(const std::vector<Dicom> &dicoms, Image *const im) {
 	int i;
 	// Check that the files are from the same series
	const int num_files = dicoms.size();
	const Dicom &first = dicoms[0];
	for (int i = 1; i < num_files; i++) {
 		const Dicom &dicom = dicoms[i];
 		if (!first.eqSeries(dicom)) {
			SIFT3D_ERR("read_dcm_dir_cpp: file %s is from a "
				"different series than file %s \n",
				dicom.name(), first.name());
			return SIFT3D_FAILURE;
		}
	}
 	// Initialize the output dimensions
	int nx = first.getNx();
	int ny = first.getNy();
	int nc = first.getNc();
 	// Verify the dimensions of the other files, counting the total
	// series z-dimension
	int nz = 0;
	for (i = 0; i < num_files; i++) {
 		// Get a slice
		const Dicom &dicom = dicoms[i];
 		// Verify the dimensions
		if (dicom.getNx() != nx || dicom.getNy() != ny ||
			dicom.getNc() != nc) {
			SIFT3D_ERR("read_dcm_dir_cpp: slice %s "
				"(%d, %d, %d) does not match the "
				"dimensions of slice %s (%d, %d, %d) \n",
				dicom.name(), dicom.getNx(),
				dicom.getNy(), dicom.getNc(),
				first.name(), nx, ny, nc);
			return SIFT3D_FAILURE;
		}
 		// Count the z-dimension
		nz += dicom.getNz();
	}
 	// Resize the output
	im->nx = nx;
	im->ny = ny;
	im->nz = nz;
	//im->nc = nc;
	im->ux = first.getUx();
	im->uy = first.getUy();
	im->uz = first.getUz();
	//im_default_stride(im);
 	im->xs = 1;
	im->ys = nx;
	im->zs = nx * ny;
	if (im_resize(im))
		return SIFT3D_FAILURE;
 	//Allocate memory here
	/*const size_t size = im->nx * im->ny * im->nz;
	if (im->size == size)
	return SIFT3D_SUCCESS;
 	im->size = size;
	hcreateptr(im->data, size * sizeof(float));*/
 	return SIFT3D_SUCCESS;
}
 /* Helper function to read DICOM image data */
static int read_dcm_img(const Dicom &dicom, Image *const im) {
 	const void *data;
	const DiMonoPixel *pixels;
	uint32_t shift;
	int x, y, z, depth;
	const int bufNBits = 32;
 	// Initialize JPEG decoders
	DJDecoderRegistration::registerCodecs();
 	// Initialize the DicomImage object
	const char *path = dicom.name();
	DicomImage dicomImage(path);
	if (dicomImage.getStatus() != EIS_Normal) {
		SIFT3D_ERR("read_dcm_cpp: failed to open image %s (%s)\n",
			path, DicomImage::getString(dicomImage.getStatus()));
		goto read_dcm_img_quit;
	}
 	// Initialize the image fields
	im->nx = dicom.getNx();
	im->ny = dicom.getNy();
	im->nz = dicom.getNz();
	//im->nc = dicom.getNc();
	im->ux = dicom.getUx();
	im->uy = dicom.getUy();
	im->uz = dicom.getUz();
 	// Resize the output
	//im_default_stride(im);
	im->xs = 1;
	im->ys = im->nx;
	im->zs = im->nx * im->ny;
	if (im_resize(im))
		goto read_dcm_img_quit;
 	// Get the vendor-independent intermediate pixel data
	pixels = (const DiMonoPixel *)dicomImage.getInterData();
	if (pixels == NULL) {
		SIFT3D_ERR("read_dcm_img: failed to get intermediate data for "
			"%s\n", path);
		goto read_dcm_img_quit;
	}
	depth = pixels->getBits();
 	// Macro to copy the data
#define COPY_DATA(type) \
        SIFT3D_IM_LOOP_START(im, x, y, z) \
                const int y_stride = im->nx; \
                const int z_stride = im->nx * im->ny; \
                SIFT3D_IM_GET_VOX(im, x, y, z) = \
                        (float) *((type *) data + x + y * y_stride + \
                                z * z_stride);\
        SIFT3D_IM_LOOP_END
 	// Choose the appropriate data type and copy the data
	data = pixels->getData();
 	switch (pixels->getRepresentation()) {
	case EPR_Uint8:
		COPY_DATA(uint8_t)
			break;
	case EPR_Uint16:
		COPY_DATA(uint16_t)
			break;
	case EPR_Uint32:
		COPY_DATA(uint32_t)
			break;
	case EPR_Sint8:
		COPY_DATA(int8_t)
			break;
	case EPR_Sint16:
		COPY_DATA(int16_t)
			break;
	case EPR_Sint32:
		COPY_DATA(int32_t)
			break;
	default:
		SIFT3D_ERR("read_dcm_img: unrecognized pixel representation "
			"for %s\n", path);
		goto read_dcm_img_quit;
	}
#undef COPY_DATA
 	// Clean up
	DJDecoderRegistration::cleanup();
 	return SIFT3D_SUCCESS;
 read_dcm_img_quit:
	DJDecoderRegistration::cleanup();
	return SIFT3D_FAILURE;
}
 /* Helper function to copy a slice into a larger volume. */
static int copy_slice(const Image *const slice, const int off_z,
	Image *const volume) {
 	int x, y, z;
 	// Verify dimensions
	if (volume->nz < slice->nz + off_z ||
		volume->nx != slice->nx ||
		volume->ny != slice->ny)
		return SIFT3D_FAILURE;
 	// Copy the data
	SIFT3D_IM_LOOP_START(slice, x, y, z)
 		SIFT3D_IM_GET_VOX(volume, x, y, z + off_z) =
		SIFT3D_IM_GET_VOX(slice, x, y, z);
 	SIFT3D_IM_LOOP_END
 		return SIFT3D_SUCCESS;
}
 int im_resize(Image *const im)
{
	//FIXME: This will not work for strange strides
	const size_t size = im->nx * im->ny * im->nz;
 	// Do nothing if the size has not changed
	if (im->size == size)
		return SIFT3D_SUCCESS;
	im->size = size;
 	// Allocate new memory
	CMemManager<float>::hcreateptr(im->data, size);
 	return size != 0 && im->data == NULL ? SIFT3D_FAILURE : SIFT3D_SUCCESS;
}