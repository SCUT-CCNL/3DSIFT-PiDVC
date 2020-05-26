
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#include "../Include/MemManager.h"
#include "../Include/mulDVC.h"
#include "../Include/matrixIO3D.h"
#include "../Include/conf.h"

#include <yaml-cpp/yaml.h>
#include <3DSIFT/Include/cSIFT3D.h>
#include <3DSIFT/Include/cUtil.h>
#include <3DSIFT/Include/cMatcher.h>


using namespace std;

enum SIFTMode {
	IORead = 1,
	RunAndWrite = 2,
};

static map<string, POIManager::generateType> poiGenMap = {
	{ string("cuboid"),POIManager::generateType::uniform },
	{ string("import"),POIManager::generateType::ioPreset },
};

static map<string, GuessMethod> guessMethodMap = {
	{ string("PiSIFT"), PiSIFT },
	{ string("IOPreset"), IOPreset },
	{ string("FFT-CC"), FFTCCGuess },
};

static map<string, SIFTMode> siftModeMap = {
	{ string("IORead"), IORead },
	{ string("Run"), RunAndWrite },
};

//Function for SIFT extraction and matching, or reading results stored in text file
SIFT_PROCESS Sift_(const string Ref_file_name, const string Tar_file_name, CPaDVC* sepaDVC, SIFTMode Mode);

//util func for yaml parsing
YAML::Node GetYAMLConfig(std::string path);

int main(int argc, char *argv[]){
#define VERSION_DESC "Strategy2_CPU"
	if (argc < 2) {
		std::cerr << "Too few arguments" << std::endl;
		getchar();
		return 0;
	}

	auto config = GetYAMLConfig(argv[1]);
	string importPOIPath;
	int_3d totalNum, gridSpace, subVolSize, startXYZ;

	POIManager::generateType roiType;
	GuessMethod initMethod;
	SIFTMode m_eSiftMode = IORead;
	float ransacThres = 4.0f;
	int ransacIter = 48, minNeighborNum = 16;
	int numThreads = 20, icgnMaxIter = 20;
	float icgnDeltaP = 0.001f;
	SIFT_PROCESS m_SIFT_TIME;

	//read from files
	string refImgPath = config["filepath"]["ref"]["image"].as<std::string>();
	string tarImgPath = config["filepath"]["tar"]["image"].as<std::string>();
	string refKpPath = config["filepath"]["ref"]["key_points"].as<std::string>();
	string tarKpPath = config["filepath"]["tar"]["key_points"].as<std::string>();
	string outputDir = config["filepath"]["result_dir"].as<std::string>() + string("\\");

	string ROIMode = config["roi"]["mode"].as<std::string>();
	roiType = getValue(poiGenMap, ROIMode);
	switch (roiType) {
	case POIManager::uniform:
		startXYZ = int_3d(
			config["roi"]["start_coor"]["x"].as<int>(),
			config["roi"]["start_coor"]["y"].as<int>(),
			config["roi"]["start_coor"]["z"].as<int>());

		totalNum = int_3d(
			config["roi"]["poi_num"]["x"].as<int>(),
			config["roi"]["poi_num"]["y"].as<int>(),
			config["roi"]["poi_num"]["z"].as<int>()
		);

		gridSpace = int_3d(
			config["roi"]["grid_space"]["x"].as<int>(),
			config["roi"]["grid_space"]["y"].as<int>(),
			config["roi"]["grid_space"]["z"].as<int>());
		break;
	case POIManager::ioPreset:
		importPOIPath = config["filepath"]["poi_coor"].as<std::string>();
		break;
	}
	subVolSize = int_3d(
		config["roi"]["subset_radius"]["x"].as<int>(),
		config["roi"]["subset_radius"]["y"].as<int>(),
		config["roi"]["subset_radius"]["z"].as<int>());

	//initial parameters
	string sInitMethodStr = config["dvc"]["initial"]["method"].as<std::string>();
	initMethod = getValue(guessMethodMap, sInitMethodStr);
	switch (initMethod) {
	case PiSIFT:
		ransacThres = config["dvc"]["initial"]["ransac_error"].as<float>();
		ransacIter = config["dvc"]["initial"]["ransac_max_iter"].as<int>();
		minNeighborNum = config["dvc"]["initial"]["min_neighbor"].as<int>();
		m_eSiftMode = getValue(siftModeMap, config["filepath"]["sift_mode"].as<std::string>());
		break;
	case FFTCC:
		break;
	case IOPreset:
		break;
	default:
		break;
	};

	icgnMaxIter = config["dvc"]["iterative"]["icgn_max_iter"].as<int>();
	icgnDeltaP = config["dvc"]["iterative"]["icgn_deltaP"].as<float>();
	numThreads = config["dvc"]["num_thread"].as<int>();

	auto t_ = time(nullptr);
	auto tm_ = *localtime(&t_);
	stringstream sParameter;
	string name_(tarImgPath.begin() + tarImgPath.find_last_of('\\') + 1, tarImgPath.begin() + tarImgPath.find_last_of('.'));
	sParameter << name_ << "_" << VERSION_DESC << "_" << sInitMethodStr;
	if (initMethod == PiSIFT) {
		sParameter << "_" << "N" << minNeighborNum << "_" << "Ransac" << ransacIter << "_" << ransacThres;
	}
	sParameter << "_" << "Sub" << 2 * subVolSize.x + 1 << put_time(&tm_, "_%Y-%m-%d-%H-%M");

	string fileName = sParameter.str() + string(".txt");
	string resultPath = outputDir + fileName;

	string crefPath = string(""), ctarPath = string(""), frefPath = string(""), ftarPath = string("");
	if (initMethod == PiSIFT) {
		//crefPath = outputDir + string("cref_") + fileName;
		//ctarPath = outputDir + string("ctar_") + fileName;
		//frefPath = outputDir + string("fref_") + fileName;
		//ftarPath = outputDir + string("ftar_") + fileName;
	}

	//read image and set DVC
	CPaDVC* sepaDVC = nullptr;
	sepaDVC = CCUPaDVCFactory::CreateCUPaDVC(refImgPath, tarImgPath, subVolSize, numThreads);

	//set POI
	switch (roiType) {
	case POIManager::uniform:
		sepaDVC->m_POI_global = POIManager::uniformGenerator(startXYZ, int_3d{1,1,1}, gridSpace, totalNum);
		POIManager::uniformNeighborAssign(sepaDVC->m_POI_global, totalNum, numThreads);
		break;
	case POIManager::ioPreset:
		if (initMethod == IOPreset) {
			sepaDVC->m_POI_global = POIManager::ioGenerator(importPOIPath, int_3d{ 1,1,1 }, true);
		}
		else {
			sepaDVC->m_POI_global = POIManager::ioGenerator(importPOIPath, int_3d{ 1,1,1 });
			sepaDVC->KD_NeighborPOI_assign_6NN();
		}
		break;
	}

	//SIFT
	if(initMethod == PiSIFT)
		m_SIFT_TIME = Sift_(refKpPath, tarKpPath, sepaDVC, m_eSiftMode);

	//DVC Part
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	std::cout << "Start DVC part" << std::endl;

	//set Param and run 
	sepaDVC->SetICGNParam(icgnDeltaP, icgnMaxIter);
	switch (initMethod) {
	case PiSIFT:
		sepaDVC->SetPiSIFTInitParam(minNeighborNum, ransacThres, ransacIter);
		sepaDVC->PiDVCAlgorithm_mul_global_STRATEGY(); //main funcs
		break;
	case FFTCCGuess:
		sepaDVC->PiDVCAlgorithm_mul_global_FFTCC(); //main funs
		break;
	case IOPreset:
		sepaDVC->PiDVCAlgorithm_mul_global_ICGN_Only(); //main funs
		break;
	}

	std::chrono::system_clock::time_point ed = std::chrono::system_clock::now();
	std::chrono::duration<double> duration = ed - start;
	std::cout << "Finish" << std::endl;
	std::cout << "Total duration[s]:\t" << duration.count() << "" << std::endl;
	std::cout << "Press enter to continue..." << std::endl;

	sepaDVC->SaveResult2Text_global_d(resultPath, m_SIFT_TIME, initMethod, crefPath,ctarPath, frefPath,	ftarPath);

	delete sepaDVC;
	return 0;
}

SIFT_PROCESS Sift_(const string Ref_file_name, const string Tar_file_name, CPaDVC* sepaDVC, SIFTMode Mode) {

    SIFT_PROCESS m_SIFT_TIME;
    clock_t start, end;

    if (Mode == RunAndWrite) {

        start = clock();

        auto Ref_SIFT3D = CPUSIFT::CSIFT3DFactory::CreateCSIFT3D(sepaDVC->m_fVolR, sepaDVC->m_iOriginVolWidth, sepaDVC->m_iOriginVolHeight, sepaDVC->m_iOriginVolDepth);
        Ref_SIFT3D->KpSiftAlgorithm();
        m_SIFT_TIME.REF = Ref_SIFT3D->m_timer;
        auto RefKp = Ref_SIFT3D->GetKeypoints();
        std::cout << "Reference Done, features num:\t" << RefKp.size() << std::endl;

        //TAR SIFT EXTRACT
        auto Tar_SIFT3D = CPUSIFT::CSIFT3DFactory::CreateCSIFT3D(sepaDVC->m_fVolT, sepaDVC->m_iOriginVolWidth,
            sepaDVC->m_iOriginVolHeight, sepaDVC->m_iOriginVolDepth);
        Tar_SIFT3D->KpSiftAlgorithm();
        m_SIFT_TIME.TAR = Tar_SIFT3D->m_timer;
        auto TarKp = Tar_SIFT3D->GetKeypoints();
        std::cout << "Target Done, features num:\t" << TarKp.size() << std::endl;

        //MATCH
        double reg_start = omp_get_wtime();
        CPUSIFT::muBruteMatcher matcher;
        matcher.enhancedMatch(sepaDVC->Ref_point, sepaDVC->Tar_point, RefKp, TarKp);
        double reg_end = omp_get_wtime();
        m_SIFT_TIME.d_RegTime = reg_end - reg_start;

        std::cout << "RegSift with nnmatch Done" << std::endl;
        std::cout << "Write kp_list" << std::endl;

        end = clock();
        double duration = ((double)end - start) / CLOCKS_PER_SEC;
        std::cout << "SIFT: " << duration << " seconds" << std::endl;

        CPUSIFT::write_sift_kp(sepaDVC->Ref_point, Ref_file_name.c_str());
        CPUSIFT::write_sift_kp(sepaDVC->Tar_point, Tar_file_name.c_str());

        {
            //log time
            string timeLogFile = Ref_file_name + string(".log");
            ofstream oFile;
            oFile.open(timeLogFile, ios::out | ios::trunc);
            oFile << m_SIFT_TIME << endl;
            oFile.close();
        }

        delete Ref_SIFT3D;
        delete Tar_SIFT3D;
    }
    else if (Mode == IORead) {
        CPUSIFT::read_sift_kp(Ref_file_name.c_str(), sepaDVC->Ref_point);
        CPUSIFT::read_sift_kp(Tar_file_name.c_str(), sepaDVC->Tar_point);
    }
    return m_SIFT_TIME;
}

YAML::Node GetYAMLConfig(std::string path) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(path);
    }
    catch (YAML::BadFile &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return config;
}

