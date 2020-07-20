using Microsoft.ML.Forecasting.GlobalTemperature.Samples;
using System;

namespace Microsoft.ML.Forecasting.GlobalTemperature
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Demo with known prices for comparison.");
            new DemoPrediction().Run();

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Demo for unknown dates.");
            
            //year, month
            new CustomPrediction().Header().Run(2020, 7);
            new CustomPrediction().Run(2030, 7);
            new CustomPrediction().Run(2040, 7);
            new CustomPrediction().Run(2050, 7);
        }


    }

}