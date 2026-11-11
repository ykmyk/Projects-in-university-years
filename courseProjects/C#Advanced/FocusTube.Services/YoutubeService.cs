using Google.Apis.Services;
using Google.Apis.YouTube.v3;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace FocusTube.Services
{
    /// <summary>
    /// Service responsible for communicating with the external Google YouTube Data API v3.
    /// </summary>
    public class YouTubeIntegrationService : IYouTubeIntegrationService
    {
        private readonly string _apiKey;

        /// <summary>
        /// Initializes the service with the required Google Cloud API Key.
        /// </summary>
        /// <param name="apiKey">The API key loaded from the application configuration.</param>
        public YouTubeIntegrationService(string apiKey)
        {
            _apiKey = apiKey;
        }

        /// <summary>
        /// Asynchronously fetches video metadata from Google's servers.
        /// </summary>
        /// <param name="videoId">The unique YouTube video ID.</param>
        /// <returns>A tuple containing the video's duration, category ID, live status, and channel name.</returns>
        /// <exception cref="Exception">Thrown when the API returns a null result for the provided ID.</exception>
        public async Task<(TimeSpan Duration, string CategoryId, bool IsLive, string ChannelName)> GetVideoDetailsAsync(string videoId)
        {
            var youtubeService = new YouTubeService(new BaseClientService.Initializer()
            {
                ApiKey = _apiKey,
                ApplicationName = "FocusTube"
            });

            var searchRequest = youtubeService.Videos.List("snippet,contentDetails,liveStreamingDetails");
            searchRequest.Id = videoId;

            var response = await searchRequest.ExecuteAsync();
            var video = response.Items.FirstOrDefault();
            
            if (video == null) throw new Exception("Video not found!");

            var duration = System.Xml.XmlConvert.ToTimeSpan(video.ContentDetails.Duration);
            var categoryId = video.Snippet.CategoryId;
            var isLive = video.Snippet.LiveBroadcastContent != "none";
            var channelName = video.Snippet.ChannelTitle;

            return (duration, categoryId, isLive, channelName);
        }
    }
}