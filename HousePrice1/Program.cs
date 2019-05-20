using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePrice1
{
    class Program
    {
        private static string BinPath => new FileInfo(typeof(Program).Assembly.Location).Directory?.FullName ?? string.Empty;
        private static readonly string ModelFile = Path.Combine(BinPath, @"MLModels.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(0);

            // 1. Import or create training data
            var houseData = new List<HouseData>
            {
                new HouseData {Size = 1.1F, Price = 1.2F},
                new HouseData {Size = 1.9F, Price = 2.3F},
                new HouseData {Size = 2.8F, Price = 3.0F},
                new HouseData {Size = 3.4F, Price = 3.7F}
            };
            var trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // 2. Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "Size")
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // 3. Train model
            var model = pipeline.Fit(trainingData);
            mlContext.Model.Save(model, trainingData.Schema, ModelFile);

            // 4. Make a prediction
            var size = new HouseData { Size = 2.5F };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);
            
            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k");
            Console.ReadKey();

            // testing & get metrics
            var testHouseData = new List<HouseData>
            {
                new HouseData { Size = 1.1F, Price = 0.98F },
                new HouseData { Size = 1.9F, Price = 2.1F },
                new HouseData { Size = 2.8F, Price = 2.9F },
                new HouseData { Size = 3.4F, Price = 3.6F }
            };

            var testHouseDataView = mlContext.Data.LoadFromEnumerable(testHouseData);
            var testPriceDataView = model.Transform(testHouseDataView);

            var metrics = mlContext.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");
            Console.ReadKey();
        }

        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }
    }
}
