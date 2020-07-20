using Microsoft.ML.Forecasting.GlobalTemperature.Engine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Samples
{
    public abstract class AbstractPrediction
    {
        protected DataLoader DataLoader { get; private set; }

        protected MLContext MLContext { get; private set; }

        public AbstractPrediction()
        {
            
        }

        protected void Prepare(DateTime testSegment)
        {
            MLContext = new MLContext();

            var rootDir = AppDomain.CurrentDomain.BaseDirectory;
            var csvFilePath = Path.Combine(rootDir, "Data", "GlobalTemperatures.csv");            
            DataLoader = new DataLoader(
                    MLContext,
                    csvFilePath,
                    testSegment
            );
        }
    }
}
