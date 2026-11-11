using System;

namespace FocusTube.Services.Models
{
    /// <summary>
    /// Represents a historical record of a video that was authorized and watched by the user.
    /// Stored in memory to be used for LINQ analytics.
    /// </summary>
    public class VideoLog
    {
        /// <summary>
        /// The unique 11-character number from YouTube for identifying the video.
        /// </summary>
        public string VideoId { get; set; } = string.Empty;
        
        /// <summary>
        /// The mapped human-readable category of the video.
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// The display name of the channel who is uploading the video.
        /// </summary>
        public string ChannelName { get; set; } = string.Empty;
        
        /// <summary>
        /// The exact duration of the video, deducted from the user's quota.
        /// </summary>
        public TimeSpan Duration { get; set; }
        
        /// <summary>
        /// The precise timestamp of when the video was requested and authorized.
        /// </summary>
        public DateTime WatchedAt { get; set; }
    }
}