using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using System;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Engine
{
   public class DataLoader
    {
        /// <summary>
        /// The entire dataset
        /// </summary>
        public IDataView SeriesData { get; private set; }

        /// <summary>
        /// Part of the data for training. 
        /// </summary>
        public IDataView TrainData { get; private set; }

        /// <summary>
        /// Part of the data for test and evaluation
        /// </summary>
        public IDataView TestData { get; private set; }

        /// <summary>
        /// Loads data from the given <paramref name="csvFilePath"/> file 
        /// and segments them to train and test data based on the <paramref name="testSegment"/> datetime.
        /// </summary>
        /// <param name="context">The common context for all ML.NET operations.</param>
        /// <param name="csvFilePath">The path to the CSV file that contains the data compatible with the <see cref="ModelInput"/>.</param>
        /// <param name="testSegment">A datetime to be used for segmenting the data in to two. First part to train the engine and second to test and evaluate it.</param>
        public DataLoader(MLContext context, string csvFilePath, DateTime testSegment)
        {
            SeriesData = context.Data.LoadFromTextFile<ModelInput>(csvFilePath, separatorChar: ',', hasHeader: true);
            TrainData = context.Data.FilterByCustomPredicate<ModelInput>(SeriesData, x => x.Date >= testSegment);
            TestData = context.Data.FilterByCustomPredicate<ModelInput>(SeriesData, x => x.Date < testSegment);
        }
    }
}
