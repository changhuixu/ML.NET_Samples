using System;
using System.IO;
using Microsoft.ML;

namespace IrisFlowerClustering
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        private static void Main()
        {
            var mlContext = new MLContext(seed: 0);
            var dataView = mlContext.Data.LoadFromTextFile<IrisData>(DataPath, hasHeader: false, separatorChar: ',');

            const string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            var model = pipeline.Fit(dataView);
            using (var fileStream = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            var setosa = new IrisData
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            };
            var prediction = predictor.Predict(setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");

            Console.ReadKey();
        }
    }
}
