using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TaxiFarePrediction.DataStructures;

namespace TaxiFarePrediction.Utils
{
    public class TaxiTripCsvReader
    {
        public static IEnumerable<TaxiTrip> GetDataFromCsv(string dataLocation)
        {
            return File.ReadAllLines(dataLocation)
                    .Skip(1)
                    .Select(x => x.Split(','))
                    .Select(x => new TaxiTrip
                    {
                        VendorId = x[0].Trim(),
                        RateCode = x[1].Trim(),
                        PassengerCount = float.Parse(x[2], CultureInfo.InvariantCulture),
                        TripTime = float.Parse(x[3], CultureInfo.InvariantCulture),
                        TripDistance = float.Parse(x[4], CultureInfo.InvariantCulture),
                        PaymentType = x[5].Trim(),
                        FareAmount = float.Parse(x[6], CultureInfo.InvariantCulture)
                    });

        }
    }
}
