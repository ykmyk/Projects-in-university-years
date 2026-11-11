using Microsoft.AspNetCore.Mvc;
using FocusTube.Services;
using System.Threading.Tasks;
using System;

namespace FocusTube.API.Controllers
{
    /// <summary>
    /// API Controller that is responsible for handling video watch requests and checking them
    /// against the limit with WatchGuardService and YouTubeIntegrationService.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class VideoController : ControllerBase
    {
        private readonly WatchGuardService _guard;
        private readonly IYouTubeIntegrationService _youtube;

        

        /// <summary>
        /// Initializes a new instance of the VideoController with 
        /// WatchGuardService and YouTubeIntegrationService.
        /// </summary>
        /// <param name="guard">The instance of the WatchGuardService.</param>
        /// <param name="youtube">Theinstance of the YouTubeIntegrationService.</param>
        public VideoController(WatchGuardService guard, IYouTubeIntegrationService youtube)
        {
            _guard = guard;
            _youtube = youtube;
        }

        /// <summary>
        /// Evaluate a YouTube video ID by checking up with the Google API (external)
        /// and also with the internal WatchGuardService.
        /// </summary>
        /// <param name="videoId">The 11-character YouTube video ID.</param>
        /// <returns>A HTTP response indicating whether the video is Allowed (200 OK) 
        /// or Blocked (400 Bad Request).</returns>
        [HttpGet("check/{videoId}")]
        public async Task<IActionResult> CheckVideo(string videoId)
        {
            try 
            {
                (TimeSpan duration, string categoryId, bool isLive, string channelName) = await _youtube.GetVideoDetailsAsync(videoId);

                var (categoryName, isStudy) = CategoryMapper.GetCategoryInfo(categoryId);

                if (isStudy)
                {
                    return Ok(new { status = "Allowed", reason = "Educational", category = categoryName });
                }

                var result = _guard.AuditVideo(videoId, isLive, duration, categoryName, channelName);

                if (!result.IsAllowed)
                {
                    return BadRequest(new { status = "Blocked", reason = result.Reason, category = categoryName });
                }

                return Ok(result);
            }
            catch (Exception)
            {
                return NotFound(new { message = "Video not found. Please check the ID." });
            }
        }
    }
}