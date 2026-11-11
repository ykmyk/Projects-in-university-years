using FocusTube.Core;
using Xunit;

namespace FocusTube.Tests
{
    /// <summary>
    /// Contains unit tests for the core TimeQuota domain model.
    /// Focuses on verifying the mathematical computation of the overloaded operators.
    /// </summary>
    public class TimeQuotaTests
    {
        /// <summary>
        /// Verifies that the overloaded subtraction (-) operator correctly deducts 
        /// a given TimeSpan from the internal RemainingTime of the quota.
        /// </summary>
        [Fact]
        public void OperatorMinus_ShouldSubtractDurationCorrectly()
        {
            var quota = new TimeQuota(TimeSpan.FromMinutes(60));
            var videoDuration = TimeSpan.FromMinutes(15);

            var result = quota - videoDuration;

            Assert.Equal(TimeSpan.FromMinutes(45), result.RemainingTime);
        }

        /// <summary>
        /// Verifies that the overloaded less-than (<) operator correctly identifies 
        /// when a requested video duration is longer than the quota's remaining time.
        /// </summary>
        [Fact]
        public void OperatorLessThan_ShouldReturnTrue_WhenQuotaIsTooLow()
        {
            var quota = new TimeQuota(TimeSpan.FromMinutes(5));
            var longVideo = TimeSpan.FromMinutes(10);

            Assert.True(quota < longVideo);
        }
    }
}