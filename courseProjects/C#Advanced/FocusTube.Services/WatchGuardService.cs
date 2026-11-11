using System;
using System.Collections.Generic;
using System.Linq;
using FocusTube.Core;
using FocusTube.Services.Models;

namespace FocusTube.Services
{
    /// <summary>
    /// Delegate defining the signature for handling limit exceeded events.
    /// </summary>
    /// <param name="message">The specific reason the limit was exceeded.</param>
    /// <param name="category">The category of the video that triggered the limit.</param>
    public delegate void LimitExceededHandler(string message, string category);

    /// <summary>
    /// The core domain service responsible for enforcing hierarchical time quotas, 
    /// auditing video watch requests, and tracking daily user history.
    /// </summary>
    public class WatchGuardService
    {
        /// <summary>
        /// Event triggered whenever a user attempts to watch a video that exceeds their allotted time quota.
        /// </summary>
        public event LimitExceededHandler? OnLimitExceeded;

        private TimeQuota _totalQuota = new TimeQuota(TimeSpan.FromHours(2));
        private readonly Dictionary<string, TimeQuota> _categoryQuotas = new();
        private readonly Dictionary<string, TimeQuota> _channelQuotas = new();
        private readonly List<VideoLog> _history = new List<VideoLog>();

        /// <summary>
        /// Audits a requested YouTube video against the user's hierarchical time quotas.
        /// It checks the total daily limit, the specific category limit, and the specific channel limit.
        /// </summary>
        /// <param name="videoId">The unique YouTube identifier for the video.</param>
        /// <param name="isLive">Boolean flag indicating if the video is an active live stream.</param>
        /// <param name="duration">The parsed length of the video.</param>
        /// <param name="category">The mapped human-readable category name.</param>
        /// <param name="channelName">The name of the channel uploading the video.</param>
        /// <returns>A VideoAuditResult object detailing if the video is allowed or blocked, along with the specific reason.</returns>
        public VideoAuditResult AuditVideo(string videoId, bool isLive, TimeSpan duration, string category, string channelName)
        {
            if (isLive) return new VideoAuditResult { IsAllowed = true, Reason = "Live streams are allowed." };

            var exceededReasons = new List<string>();

            // Utilizing overloaded '<' operator from TimeQuota class
            if (_totalQuota < duration) 
                exceededReasons.Add("the Total limit for today");
                
            if (_categoryQuotas.TryGetValue(category, out var catQuota) && catQuota < duration) 
                exceededReasons.Add($"the '{category}' category");
                
            if (_channelQuotas.TryGetValue(channelName, out var chanQuota) && chanQuota < duration) 
                exceededReasons.Add($"the '{channelName}' channel");

            if (exceededReasons.Count > 0)
            {
                string reasonMessage = exceededReasons.Count switch
                {
                    1 => $"exceeded the limit for {exceededReasons[0]}",
                    2 => $"exceeded the limit for {exceededReasons[0]} and also {exceededReasons[1]}",
                    _ => $"exceeded the limit for {exceededReasons[0]}, {exceededReasons[1]}, and also {exceededReasons[2]}"
                };

                OnLimitExceeded?.Invoke(reasonMessage, category);
                return new VideoAuditResult { IsAllowed = false, Reason = reasonMessage, Category = category };
            }

            // Utilizing overloaded '-' operator from TimeQuota class
            _totalQuota -= duration;
            if (_categoryQuotas.ContainsKey(category)) _categoryQuotas[category] -= duration;
            if (_channelQuotas.ContainsKey(channelName)) _channelQuotas[channelName] -= duration;

            _history.Add(new VideoLog { VideoId = videoId, Duration = duration, Category = category, ChannelName = channelName, WatchedAt = DateTime.Now });

            return new VideoAuditResult { IsAllowed = true, Title = "Authorized", Category = category };
        }

        /// <summary>
        /// Validates that a proposed time limit does not exceed the physical limits of a 24-hour day.
        /// </summary>
        /// <param name="limit">The TimeSpan to validate.</param>
        /// <exception cref="ArgumentException">Thrown when the limit is greater than 24 hours.</exception>
        private void ValidateLimit(TimeSpan limit)
        {
            if (limit.TotalHours > 24)
            {
                throw new ArgumentException("A time limit cannot exceed 24 hours per day.");
            }
        }

        /// <summary>Updates the global daily time limit.</summary>
        public void UpdateTotalLimit(TimeSpan newLimit) { ValidateLimit(newLimit); _totalQuota = new TimeQuota(newLimit); }
        
        /// <summary>Updates the time limit for a specific content category.</summary>
        public void UpdateCategoryLimit(string category, TimeSpan newLimit) { ValidateLimit(newLimit); _categoryQuotas[category] = new TimeQuota(newLimit); }
        
        /// <summary>Updates the time limit for a specific YouTube channel.</summary>
        public void UpdateChannelLimit(string channelName, TimeSpan newLimit) { ValidateLimit(newLimit); _channelQuotas[channelName] = new TimeQuota(newLimit); }

        /// <summary>Retrieves the remaining global time limit.</summary>
        public TimeSpan GetTotalRemaining() => _totalQuota.RemainingTime;
        
        /// <summary>Retrieves a dictionary of all active category limits.</summary>
        public Dictionary<string, TimeSpan> GetCategoryLimits() => _categoryQuotas.ToDictionary(k => k.Key, v => v.Value.RemainingTime);
        
        /// <summary>Retrieves a dictionary of all active channel limits.</summary>
        public Dictionary<string, TimeSpan> GetChannelLimits() => _channelQuotas.ToDictionary(k => k.Key, v => v.Value.RemainingTime);
        
        /// <summary>
        /// Utilizes LINQ to group and sum the user's daily watch history by category.
        /// </summary>
        /// <returns>A dictionary mapping category names to total time spent.</returns>
        public Dictionary<string, TimeSpan> GetDailyUsageStats() => _history.GroupBy(v => v.Category).ToDictionary(g => g.Key, g => TimeSpan.FromTicks(g.Sum(v => v.Duration.Ticks)));

        /// <summary>
        /// Utilizes LINQ to group and sum the user's daily watch history by channel.
        /// </summary>
        /// <returns>A dictionary mapping channel names to total time spent.</returns>
        public Dictionary<string, TimeSpan> GetChannelUsageStats() => _history.GroupBy(v => v.ChannelName).ToDictionary(g => g.Key, g => TimeSpan.FromTicks(g.Sum(v => v.Duration.Ticks)));
    }
}