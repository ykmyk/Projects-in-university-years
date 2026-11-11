using Microsoft.AspNetCore.Mvc;
using FocusTube.Services;
using System.Linq;
using System;
using System.Collections.Generic;

namespace FocusTube.API.Controllers
{
    /// <summary>
    /// API Controller that is responsible for aggregating and formatting the user's watcing 
    /// history and time quota limits.
    /// Provides data for the front-end Analytics Dashboard(mainly in analytics.html).
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class AnalyticsController : ControllerBase
    {
        private readonly WatchGuardService _guard;

        /// <summary>
        /// Initialize a new instance of the AnalyticsController.
        /// </summary>
        /// <param name="guard"> 
        /// WatchGuardService guard will handle the domain logic for analytics.
        /// </param>
        public AnalyticsController(WatchGuardService guard) { _guard = guard; }

        /// <summary>
        /// Retrieve a comprehensive dataset of the user's daily usage with their limits.
        /// Merge the data for total, category, and channel into a clean structure for the UI.
        /// </summary>
        /// <returns>
        /// A JSON object that contains formatted string representations of 
        /// user's time spent and limits.
        /// </returns>
        [HttpGet("summary")]
        public IActionResult GetSummary()
        {
            var catUsage = _guard.GetDailyUsageStats();
            var chanUsage = _guard.GetChannelUsageStats();
            var catLimits = _guard.GetCategoryLimits();
            var chanLimits = _guard.GetChannelLimits();
            
            var totalRemaining = _guard.GetTotalRemaining();
            var totalSpent = TimeSpan.FromTicks(catUsage.Values.Sum(ts => ts.Ticks));
            var totalLimit = totalSpent + totalRemaining; 

            // Helper function to format the output cleanly for HTML display, 
            //  showing "No Limit" when appropriate(as default).
            object BuildStat(TimeSpan spent, TimeSpan? limit) => new {
                Spent = spent.ToString(@"hh\:mm\:ss"),
                Limit = limit.HasValue ? limit.Value.ToString(@"hh\:mm\:ss") : "No Limit"
            };

            // A helper function to calculate limits 
            TimeSpan? CalculateLimit(string key, Dictionary<string, TimeSpan> usage, Dictionary<string, TimeSpan> limits)
            {
                if (!limits.ContainsKey(key)) return null; // No limit was set by the user
                
                var used = usage.ContainsKey(key) ? usage[key] : TimeSpan.Zero;
                return used + limits[key]; // Original Limit = Time Used + Time Remaining
            }

            // Merge Categories
            var allCats = catUsage.Keys.Union(catLimits.Keys).Distinct();
            var catDict = allCats.ToDictionary(
                cat => cat,
                cat => BuildStat(
                    catUsage.ContainsKey(cat) ? catUsage[cat] : TimeSpan.Zero,
                    CalculateLimit(cat, catUsage, catLimits)
                )
            );

            // Merge Channels
            var allChans = chanUsage.Keys.Union(chanLimits.Keys).Distinct();
            var chanDict = allChans.ToDictionary(
                chan => chan,
                chan => BuildStat(
                    chanUsage.ContainsKey(chan) ? chanUsage[chan] : TimeSpan.Zero,
                    CalculateLimit(chan, chanUsage, chanLimits)
                )
            );

            return Ok(new
            {
                Total = BuildStat(totalSpent, totalLimit),
                Categories = catDict,
                Channels = chanDict
            });
        }
    }
}