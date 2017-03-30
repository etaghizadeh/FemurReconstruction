//
//  main.cpp
//  FemurReconstruction
//  Using this code, it is possible to predict the full shape of bone, by having a part of it by using statistical shape model and few landmarks. To use this code you should first compile statismo ITK and VTK
//  Created by Elham Taghizadeh on 03/03/16.
//  Copyright (c) 2016 ISTB. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <map>

#include <vtkVersion.h>
#include <vtkLandmarkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkVertexGlyphFilter.h>
#include <boost/scoped_ptr.hpp>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkVersion.h>
#include <vtkTransform.h>
#include <vtkStandardMeshRepresenter.h>

#include <itkMesh.h>
#include <itkLandmarkBasedTransformInitializer.h>
#include <itkImageFileReader.h>
#include <itkPosteriorModelBuilder.h>
#include <itkStandardMeshRepresenter.h>
#include <itkStatisticalModel.h>
#include <itkStatisticalShapeModelTransform.h>
#include <itkPointsLocator.h>
#include <itkDataManager.h>
#include <itkDirectory.h>
#include <itkImage.h>
#include <itkMeshFileReader.h>
#include <itkMeshFileWriter.h>
#include <itkStandardImageRepresenter.h>



const unsigned Dimension = 3;
typedef statismo::VectorType VectorType;
typedef statismo::MatrixType MatrixType;
typedef itk::Mesh<float, Dimension> MeshType;
typedef itk::StandardMeshRepresenter<float, Dimension> RepresenterType;
typedef itk::StatisticalModel<MeshType> StatisticalModelType;
typedef itk::PointSet<float, Dimension  > PointSetType;
typedef itk::Point<double, 3> PointType;
typedef itk::VersorRigid3DTransform<double> RigidTransformType;
typedef itk::Image<float, Dimension> DistanceImageType;
typedef itk::LandmarkBasedTransformInitializer<RigidTransformType, DistanceImageType, DistanceImageType> LandmarkTransformInitializerType;
typedef itk::StatisticalShapeModelTransform<MeshType, double, Dimension> StatisticalModelTransformType;
typedef itk::PosteriorModelBuilder<MeshType> PosteriorModelBuilderType;
#if (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
typedef itk::PointsLocator< MeshType::PointsContainer > PointsLocatorType;
#else
typedef itk::PointsLocator<int, 3, double, MeshType::PointsContainer > PointsLocatorType;
#endif
typedef itk::MeshFileWriter<MeshType> DataWriterType;

vtkPolyData* loadVTKPolyData(const std::string& filename) {
    vtkPolyDataReader* reader = vtkPolyDataReader::New();
    reader->SetFileName(filename.c_str());
    reader->Update();
    vtkPolyData* pd = vtkPolyData::New();
    pd->DeepCopy(reader->GetOutput());
    reader->Delete();
    return pd;
}



class Utilities {
public:
    /**
     * read landmarks from the given file in slicer fcsv formant and return them as a list.
     *
     * The format is: label,x,y,z
     *
     * @param filename the filename
     * @returns A list of itk points
     */
    static std::vector<PointType > readLandmarks(const std::string& filename) {
        
        std::vector<PointType> ptList;
        
        std::fstream file ( filename.c_str() );
        if (!file) {
            std::cout << "could not read landmark file " << std::endl;
            throw std::runtime_error("could not read landmark file ");
        }
        std::string line;
        while (  std::getline ( file, line)) {
            if (line.length() > 0 && line[0] == '#')
                continue;
            
            std::istringstream strstr(line);
            std::string token;
            std::getline(strstr, token, ','); // ignore the label
            std::getline(strstr, token, ','); // get the x coord
            double pt0 = atof(token.c_str());
            std::getline(strstr, token, ','); // get the y coord
            double pt1 = atof(token.c_str());
            std::getline(strstr, token, ','); // get the z coord
            double pt2 = atof(token.c_str());
            PointType pt;
            pt[0] = pt0;
            pt[1] = pt1;
            pt[2] = pt2;
            ptList.push_back(pt);
        }
        return ptList;
    }
    
};

StatisticalModelType::Pointer
computePosteriorModel(const RigidTransformType* rigidTransform,
                      const StatisticalModelType* statisticalModel,
                      const  std::vector<PointType >& modelLandmarks,
                      const  std::vector<PointType >& targetLandmarks,
                      double variance) {
    
    // invert the transformand back transform the landmarks
    RigidTransformType::Pointer rinv = RigidTransformType::New();
    rigidTransform->GetInverse(rinv);
    
    StatisticalModelType::PointValueListType constraints;
    
    // We need to make sure the the points in fixed landmarks are real vertex points of the model reference.
    MeshType::Pointer reference = statisticalModel->GetRepresenter()->GetReference();
    PointsLocatorType::Pointer ptLocator = PointsLocatorType::New();
    ptLocator->SetPoints(reference->GetPoints());
    ptLocator->Initialize();
    
    assert(modelLandmarks.size() == targetLandmarks.size());
    for (unsigned i = 0; i < targetLandmarks.size(); i++) {
        
        int closestPointId = ptLocator->FindClosestPoint(modelLandmarks[i]);
        PointType refPoint = (*reference->GetPoints())[closestPointId];
        
        // compensate for the rigid transformation that was applied to the model
        PointType targetLmAtModelPos = rinv->TransformPoint(targetLandmarks[i]);
        StatisticalModelType::PointValuePairType pointValue(refPoint ,targetLmAtModelPos);
        constraints.push_back(pointValue);
        
    }
    
    PosteriorModelBuilderType::Pointer PosteriorModelBuilder = PosteriorModelBuilderType::New();
    StatisticalModelType::Pointer PosteriorModel = PosteriorModelBuilder->BuildNewModelFromModel(statisticalModel,constraints, variance, false);
    
    return PosteriorModel;
}


using namespace std;

map< string, vector<float> > readCSV(string FileName)
{
    map <string, vector<float> > Content;
    
    std::ifstream file (FileName); // file stream: mean shape landmarks
    
    while ( file.good() )
    {
        string value, key;
        
        getline ( file, key, ',' );
        vector<float> V(3, 0);
        if (!file.good()) break;
        getline ( file, value, ',' );
        V[0] = stof(value);
        getline ( file, value, ',' );
        V[1] = stof(value);
        getline ( file, value);
        V[2] = stof(value);
        Content.insert(make_pair(key, V));
        //Content[key] = V;
    }
    file.close();
    
    
    return Content;
};


void writeCSV(string FileName, map< string, vector<float> > Content)
{
    
    ofstream file; // file stream: landmarks
    
    file.open (FileName);
    for (auto itr = Content.begin(); itr!= Content.end(); itr++)
    {
        file << itr->first<<","<<itr->second[0]<<","<<itr->second[1]<<","<<itr->second[2]<<"\n";
    }
    file.close();
};


vector<float> linearRegression(vector<vector<float> > &xyCollection)
{
    int dataSize = xyCollection.size();
    double SUMx = 0;     //sum of x values
    double SUMy = 0;     //sum of y values
    double SUMxy = 0;    //sum of x * y
    double SUMxx = 0;    //sum of x^2
    double slope = 0;    //slope of regression line
    double y_intercept = 0; //y intercept of regression line
    double AVGy = 0;     //mean of y
    double AVGx = 0;     //mean of x
    
    //calculate various sums
    for (int i = 0; i < dataSize; i++)
    {
        //sum of x
        SUMx = SUMx + xyCollection[i][0];
        //sum of y
        SUMy = SUMy + xyCollection[i][1];
        //sum of squared x*y
        SUMxy = SUMxy + xyCollection[i][0] * xyCollection[i][1];
        //sum of squared x
        SUMxx = SUMxx + xyCollection[i][0] * xyCollection[i][0];
    }
    
    //calculate the means of x and y
    AVGy = SUMy / dataSize;
    AVGx = SUMx / dataSize;
    
    //slope or a1
    slope = (dataSize * SUMxy - SUMx * SUMy) / (dataSize * SUMxx - SUMx*SUMx);
    
    //y itercept or a0
    y_intercept = AVGy - slope * AVGx;
    
    
    vector<float> Result(2, 0);
    
    Result[0] = slope;
    Result[1] = y_intercept;
    return Result;
}


int main(int argc, const char * argv[])
{
    if (argc != 6)
    {
        cout << "usage " << argv[0] << " modelPath model_landmarks sampleLandmarks output patientHeight" << endl;
        exit(-1);
    }
    
    string modelName = argv[1];
    string fixFile = argv[2];
    string movFile = argv[3];
    string outputFile = argv[4];
    float Height = stof(argv[5]);
    const unsigned top = 57457, bottom = 164;
    
    map <string, vector<float> > fixLM = readCSV(fixFile);
    map <string, vector<float> > movLM = readCSV(movFile);
    
    
    // First we use rigid alignment based on corresponding landmarks on the model (source) and the test sample (target)
    vtkSmartPointer<vtkPoints> sourcePoints = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPoints> targetPoints = vtkSmartPointer<vtkPoints>::New();

    for (auto itr = fixLM.begin(); itr!= fixLM.end(); itr++)
    {
        float *Point = &(itr->second[0]);
        targetPoints->InsertNextPoint(Point);
        Point = &(movLM[itr->first][0]);
        sourcePoints->InsertNextPoint(Point);
    }
    
    // Setup the transform
    vtkSmartPointer<vtkLandmarkTransform> landmarkTransform = vtkSmartPointer<vtkLandmarkTransform>::New();
    landmarkTransform->SetSourceLandmarks(sourcePoints);
    landmarkTransform->SetTargetLandmarks(targetPoints);
    landmarkTransform->SetModeToRigidBody();
    landmarkTransform->Update(); //should this be here?
    
    vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
    source->SetPoints(sourcePoints);
    
    vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();
    target->SetPoints(targetPoints);
    
    vtkSmartPointer<vtkVertexGlyphFilter> sourceGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    #if VTK_MAJOR_VERSION <= 5
        sourceGlyphFilter->SetInputConnection(source->GetProducerPort());
    #else
        sourceGlyphFilter->SetInputData(source);
    #endif
    sourceGlyphFilter->Update();
    
    vtkSmartPointer<vtkVertexGlyphFilter> targetGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    #if VTK_MAJOR_VERSION <= 5
        targetGlyphFilter->SetInputConnection(target->GetProducerPort());
    #else
        targetGlyphFilter->SetInputData(target);
    #endif
    targetGlyphFilter->Update();
    
    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformFilter->SetInputConnection(sourceGlyphFilter->GetOutputPort());
    transformFilter->SetTransform(landmarkTransform);
    transformFilter->Update();
    
    vtkPolyData *Out = transformFilter->GetOutput();
    map <string, vector<float> > RegLM;
    
    
    string RegFile;
    int i=0;
    for (auto itr = fixLM.begin(); itr!= fixLM.end(); itr++)
    {
        double *Coords = Out->GetPoint(i++);
        RegLM[itr->first].push_back(*(Coords++));
        RegLM[itr->first].push_back(*(Coords++));
        RegLM[itr->first].push_back(*(Coords));
    }
    
    // One landmark with the label "0" is used to put the femoral heads at the same location.
    vector<float> sampleLMTop = RegLM["0"], meanLMTop = fixLM["0"], translate = sampleLMTop;
    
    translate[0] = meanLMTop[0] - sampleLMTop[0];
    translate[1] = meanLMTop[1] - sampleLMTop[1];
    translate[2] = meanLMTop[2] - sampleLMTop[2];
    
    for (auto it = RegLM.begin(); it != RegLM.end(); it++) {
        RegLM[it->first][0] = RegLM[it->first][0] + translate[0];
        RegLM[it->first][1] = RegLM[it->first][1] + translate[1];
        RegLM[it->first][2] = RegLM[it->first][2] + translate[2];
    }
    
    RegFile.append(movFile.begin(), movFile.end()-4);
    RegFile.append("_Reg.csv");
    writeCSV(RegFile, RegLM);

    
    // posterior model
    // The test bone's landmarks that are aligned to the model are used to constraine the model.
    string partialShapeMeshName = RegFile;
    string posteriorModelName;
    posteriorModelName.append(movFile.begin(), movFile.end()-4);
    posteriorModelName.append(".h5");

    std::vector<PointType> fixedLandmarks = Utilities::readLandmarks(fixFile);
    std::vector<PointType> movingLandmarks = Utilities::readLandmarks(RegFile);

    
    RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
    LandmarkTransformInitializerType::Pointer initializer = LandmarkTransformInitializerType::New();
    initializer->SetFixedLandmarks(fixedLandmarks);
    initializer->SetMovingLandmarks(movingLandmarks);
    initializer->SetTransform(rigidTransform);
    initializer->InitializeTransform();
    
    StatisticalModelType::Pointer model = StatisticalModelType::New();
    RepresenterType::Pointer representer = RepresenterType::New();
    model->Load(representer, modelName.c_str());
    
    StatisticalModelType::Pointer constraintModel = computePosteriorModel(rigidTransform, model, fixedLandmarks, movingLandmarks, 1);
    
    constraintModel->Save(posteriorModelName.c_str());
    
    //double femLength = Height * 0.2674;
    double femLength = Height * 0.2399;

    
    typename StatisticalModelType::VectorType coefficients(constraintModel->GetNumberOfPrincipalComponents());

    
    vector<vector<float> > sampleLength(7, vector<float>(2, 0));
    
    for (int i=-3; i<=3; i++)
    {
        coefficients.fill(0);
        coefficients[0] = i;// * sqrt(latents[0]);
        
        auto samplePC = constraintModel->DrawSample(coefficients);
        PointType topCoord = samplePC->GetPoint(top), btmCoord = samplePC->GetPoint(bottom);
        sampleLength[i+3][0] = sqrt(pow(topCoord[0] - btmCoord[0],2) + pow(topCoord[1] - btmCoord[1],2) + pow(topCoord[2] - btmCoord[2],2))/10;
        sampleLength[i+3][1] = i;
    }
    
    vector<float> Fit = linearRegression(sampleLength);
    double bParam = femLength * Fit[0] + Fit [1];
    
    coefficients[0] = bParam;
    StatisticalModelType::DatasetPointerType output = constraintModel->DrawSample(coefficients);
    
    
    typename DataWriterType::Pointer pDataWriter = DataWriterType::New();
    pDataWriter->SetFileName(outputFile);
    pDataWriter->SetInput(output);
    pDataWriter->Update();
    
    
    
    
    
    vtkSmartPointer<vtkPolyDataReader> vtkReader = vtkSmartPointer<vtkPolyDataReader>::New();
    vtkReader->SetFileName(outputFile.c_str());
    vtkReader->Update();
    source->DeepCopy(vtkReader->GetOutput());
    
    // first we reverse the applied translation
    vtkSmartPointer<vtkTransform> transformation = vtkSmartPointer<vtkTransform>::New();
    transformation->Translate(double(translate[0]), double(translate[1]), double(translate[2]));
    transformation->Inverse();
    transformation->Update();
    
    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter2 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    
#if VTK_MAJOR_VERSION <= 5
    transformFilter2->SetInputConnection(source->GetProducerPort());
#else
    transformFilter2->SetInputData(source);
#endif
    transformFilter2->SetTransform(transformation);
    transformFilter2->Update();
    source->DeepCopy(transformFilter2->GetOutput());
    
    
    
    // now we can apply the inverted transformation
    vtkAbstractTransform *T = transformFilter->GetTransform();
    T->Inverse();
    T->Update();
    transformFilter2 = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformFilter2->SetInputData(source);
    transformFilter2->SetTransform(T);
    transformFilter2->Update();
    
    vtkSmartPointer<vtkPolyDataWriter> vtkWriter = vtkSmartPointer<vtkPolyDataWriter>::New();
    vtkWriter->SetFileName(outputFile.c_str());
    vtkWriter->SetInputData(transformFilter2->GetOutput());
    vtkWriter->Update();
    
    return 0;
}
