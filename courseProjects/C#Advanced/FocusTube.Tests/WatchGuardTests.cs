using System;
using System.Collections.Generic;
using FocusTube.Services;
using Xunit;

namespace FocusTube.Tests
{
    /// <summary>
    /// Contains unit tests for the WatchGuardService domain logic.
    /// Tests hierarchical quota enforcement, LINQ aggregation, and event delegates as well.
    /// </summary>
    public class WatchGuardTests
    {
        /// <summary>
        /// Verifies that when a video exceeds the allowed limit, the OnLimitExceeded 
        /// event delegate is successfully invoked with the correct error message.
        /// </summary>
        [Fact]
        public void AuditVideo_ShouldTriggerDelegate_WhenTotalQuotaExceeded()
        {
            var service = new WatchGuardService();
            string? receivedMessage = null;
            service.OnLimitExceeded += (msg, cat) => receivedMessage = msg;

            var hugeVideo = TimeSpan.FromHours(3); 
            
            service.AuditVideo("test_id", false, hugeVideo, "Entertainment", "Random Channel");

            Assert.NotNull(receivedMessage);
            Assert.Contains("exceeded the limit for", receivedMessage);
        }

        /// <summary>
        /// Verifies that the internal history tracking and LINQ grouping mechanisms 
        /// accurately sum the total time spent watching videos in a specific category.
        /// </summary>
        [Fact]
        public void GetDailyUsageStats_ShouldUseLINQToSumCorrectly()
        {
            var service = new WatchGuardService();
            var duration = TimeSpan.FromMinutes(10);

            service.AuditVideo("vid1", false, duration, "Gaming", "PewDiePie");
            service.AuditVideo("vid2", false, duration, "Gaming", "PewDiePie");

            var stats = service.GetDailyUsageStats();

            Assert.Equal(TimeSpan.FromMinutes(20), stats["Gaming"]);
        }

        /// <summary>
        /// Verifies that the settings update methods successfully overwrite existing quotas
        /// across the global(total), category, and channel levels.
        /// </summary>
        [Fact]
        public void UpdateLimits_ShouldCorrectlyApplyNewHierarchicalSettings()
        {
            var service = new WatchGuardService();
            
            service.UpdateTotalLimit(TimeSpan.FromHours(5));
            service.UpdateCategoryLimit("Gaming", TimeSpan.FromMinutes(30));
            service.UpdateChannelLimit("MrBeast", TimeSpan.FromMinutes(15));

            Assert.Equal(TimeSpan.FromHours(5), service.GetTotalRemaining());
            Assert.Equal(TimeSpan.FromMinutes(30), service.GetCategoryLimits()["Gaming"]);
            Assert.Equal(TimeSpan.FromMinutes(15), service.GetChannelLimits()["MrBeast"]);
        }

        /// <summary>
        /// Ensures that a video is blocked if it violates a category quota, 
        /// even if the total quota has time remaining.
        /// </summary>
        [Fact]
        public void AuditVideo_ShouldBlock_WhenOnlyCategoryLimitIsExceeded()
        {
            var service = new WatchGuardService();
            service.UpdateCategoryLimit("Gaming", TimeSpan.FromMinutes(15));

            var result = service.AuditVideo("vid", false, TimeSpan.FromMinutes(20), "Gaming", "IGN");

            Assert.False(result.IsAllowed);
            Assert.Contains("exceeded the limit for the 'Gaming' category", result.Reason);
        }

        /// <summary>
        /// Ensures that a video is blocked if it violates a specific channel quota, 
        /// regardless of other remaining quotas.
        /// </summary>
        [Fact]
        public void AuditVideo_ShouldBlock_WhenOnlyChannelLimitIsExceeded()
        {
            var service = new WatchGuardService();
            service.UpdateChannelLimit("DistractingChannel", TimeSpan.FromMinutes(5));

            var result = service.AuditVideo("vid", false, TimeSpan.FromMinutes(10), "Entertainment", "DistractingChannel");

            Assert.False(result.IsAllowed);
            Assert.Contains("exceeded the limit for the 'DistractingChannel' channel", result.Reason);
        }

        /// <summary>
        /// Verifies that the dynamic string builder correctly combines multiple violation reasons 
        /// into a single, cohesive error message for the user interface.
        /// </summary>
        [Fact]
        public void AuditVideo_ShouldProvideCombinedMessage_WhenMultipleLimitsExceeded()
        {
            var service = new WatchGuardService();
            service.UpdateTotalLimit(TimeSpan.FromMinutes(5));
            service.UpdateCategoryLimit("Gaming", TimeSpan.FromMinutes(5));
            service.UpdateChannelLimit("BadChannel", TimeSpan.FromMinutes(5));

            var result = service.AuditVideo("vid", false, TimeSpan.FromMinutes(30), "Gaming", "BadChannel");

            Assert.False(result.IsAllowed);
            Assert.Contains("Total limit", result.Reason);
            Assert.Contains("'Gaming'", result.Reason);
            Assert.Contains("'BadChannel'", result.Reason);
        }

        /// <summary>
        /// Ensures domain validation is enforced by throwing an exception when a user attempts 
        /// to set a daily limit that exceeds the 24 hours available in a physical day.
        /// </summary>
        [Fact]
        public void UpdateLimits_ShouldThrowException_WhenLimitExceeds24Hours()
        {
            var service = new WatchGuardService();
            var invalidTime = TimeSpan.FromHours(25);

            var exception = Assert.Throws<ArgumentException>(() => 
            {
                service.UpdateTotalLimit(invalidTime);
            });

            Assert.Contains("cannot exceed 24 hours", exception.Message);
        }
    }
}