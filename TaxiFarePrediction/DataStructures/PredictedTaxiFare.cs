using Microsoft.ML.Data;

namespace TaxiFarePrediction.DataStructures
{
    public class PredictedTaxiFare
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
