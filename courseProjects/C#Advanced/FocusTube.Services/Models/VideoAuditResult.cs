using System;

namespace FocusTube.Services.Models
{
    /// <summary>
    /// Represents the outcome of an audit performed against the user's time quotas.
    /// Used to inform to the frontend if a video is allowed or blocked.
    /// </summary>
    public class VideoAuditResult
    {
        /// <summary>
        /// Indicates whether the user has sufficient time remaining to watch the given video.
        /// </summary>
        public bool IsAllowed { get; set; }
        
        /// <summary>
        /// The specific explanation of with which limit violation a video was blocked.
        /// </summary>
        public string Reason { get; set; } = string.Empty;
        
        /// <summary>
        /// The generic title assigned to the audit result.
        /// </summary>
        public string Title { get; set; } = string.Empty;
        
        /// <summary>
        /// The human-readable category mapped from the YouTube API.
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// The total duration of the audited video.
        /// </summary>
        public TimeSpan Duration { get; set; }
    }
}