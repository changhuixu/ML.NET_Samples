using System;
using System.IO;
using Microsoft.ML;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            Console.WriteLine($"=============== Loading Dataset  ===============");
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
            Console.WriteLine($"=============== Finished Loading Dataset  ===============");

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate(_trainingDataView.Schema);
            PredictIssue();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            Console.WriteLine($"=============== Processing Data ===============");
            var pipeline =
                _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                    .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                    .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                    .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                    .AppendCacheCheckpoint(_mlContext);
            Console.WriteLine($"=============== Finished Processing Data ===============");
            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // STEP 3: Create the training algorithm/trainer
            // Use the multi-class SDCA algorithm to predict the label using features.
            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine($"=============== Training the model  ===============");
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            Console.WriteLine($"=============== Finished Training the model Ending time: {DateTime.Now:O} ===============");

            // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
            Console.WriteLine($"=============== Single Prediction just-trained-model ===============");

            // Create prediction engine related to the loaded trained model
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);
            var issue = new GitHubIssue
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            // STEP 5:  Evaluate the model in order to get the model's accuracy metrics
            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Starting time: {DateTime.Now:O} ===============");

            //Load the test dataset into the IDataView
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

            //Evaluate the model on a test dataset and calculate metrics of the model on the test data.
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now:O} ===============");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            // Save the new model to .ZIP file
            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);

        }

        public static void PredictIssue()
        {
            var loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            var singleIssue = new GitHubIssue { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            //Predict label for single hard-coded issue
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
