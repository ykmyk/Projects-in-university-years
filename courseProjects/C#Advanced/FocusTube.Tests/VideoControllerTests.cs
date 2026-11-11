using System;
using System.Threading.Tasks;
using FocusTube.API.Controllers;
using FocusTube.Services;
using FocusTube.Services.Models;
using Microsoft.AspNetCore.Mvc;
using Moq;
using Xunit;

namespace FocusTube.Tests
{
    /// <summary>
    /// Contains unit tests for the VideoController, by mocking Youtube API.
    /// Utilizes the Moq library to isolate the controller from the external Youtube API,
    /// ensuring fast, reliable, and offline testing of the HTTP request pipeline.
    /// </summary>
    public class VideoControllerTests
    {
        /// <summary>
        /// Verifies that when the mocked YouTube API returns a category flagged as "Study" (e.g., Education),
        /// the controller bypasses the WatchGuard quota check and returns a 200 OK status.
        /// </summary>
        [Fact]
        public async Task CheckVideo_ShouldReturnAllowed_WhenMockYouTubeReturnsStudyCategory()
        {
            var mockGuard = new WatchGuardService();
            var mockYouTube = new Mock<IYouTubeIntegrationService>();
            
            mockYouTube.Setup(api => api.GetVideoDetailsAsync("study_123"))
                       .ReturnsAsync((TimeSpan.FromMinutes(10), "27", false, "Khan Academy"));

            var controller = new VideoController(mockGuard, mockYouTube.Object);

            var result = await controller.CheckVideo("study_123");

            var okResult = Assert.IsType<OkObjectResult>(result);
            var responseString = okResult.Value?.ToString();
            
            Assert.Contains("Allowed", responseString);
            Assert.Contains("Education", responseString);
        }

        /// <summary>
        /// Verifies that when the user has zero global quota remaining, a request to watch 
        /// a non-educational video (e.g., Gaming) results in a 400 Bad Request status.
        /// </summary>
        [Fact]
        public async Task CheckVideo_ShouldReturnBlocked_WhenMockYouTubeReturnsGamingAndQuotaIsZero()
        {
            var mockGuard = new WatchGuardService();
            mockGuard.UpdateTotalLimit(TimeSpan.Zero); 

            var mockYouTube = new Mock<IYouTubeIntegrationService>();
            mockYouTube.Setup(api => api.GetVideoDetailsAsync("game_123"))
                       .ReturnsAsync((TimeSpan.FromMinutes(30), "20", false, "PewDiePie"));

            var controller = new VideoController(mockGuard, mockYouTube.Object);

            var result = await controller.CheckVideo("game_123");

            var badRequestResult = Assert.IsType<BadRequestObjectResult>(result);
            var responseString = badRequestResult.Value?.ToString();
            
            Assert.Contains("Blocked", responseString);
            Assert.Contains("Gaming", responseString);
        }

        /// <summary>
        /// Verifies the hierarchical logic: if the total limit is fine, but the specific 
        /// category limit is exceeded, the controller must reject the request and cite the category.
        /// </summary>
        [Fact]
        public async Task CheckVideo_ShouldReturnBlocked_WhenOnlyCategoryLimitExceeded()
        {
            var mockGuard = new WatchGuardService();
            mockGuard.UpdateCategoryLimit("Gaming", TimeSpan.FromMinutes(10));

            var mockYouTube = new Mock<IYouTubeIntegrationService>();
            mockYouTube.Setup(api => api.GetVideoDetailsAsync("cat_block_123"))
                       .ReturnsAsync((TimeSpan.FromMinutes(20), "20", false, "IGN"));

            var controller = new VideoController(mockGuard, mockYouTube.Object);

            var result = await controller.CheckVideo("cat_block_123");

            var badRequestResult = Assert.IsType<BadRequestObjectResult>(result);
            var responseString = badRequestResult.Value?.ToString();
            
            Assert.Contains("Blocked", responseString);
            Assert.Contains("exceeded the limit for the 'Gaming' category", responseString);
        }

        /// <summary>
        /// Verifies the hierarchical logic: if both total and category limits are fine, 
        /// but the specific channel limit is exceeded, the controller must reject the request.
        /// </summary>
        [Fact]
        public async Task CheckVideo_ShouldReturnBlocked_WhenOnlyChannelLimitExceeded()
        {
            var mockGuard = new WatchGuardService();
            mockGuard.UpdateChannelLimit("DistractingChannel", TimeSpan.FromMinutes(5));

            var mockYouTube = new Mock<IYouTubeIntegrationService>();
            mockYouTube.Setup(api => api.GetVideoDetailsAsync("chan_block_123"))
                       .ReturnsAsync((TimeSpan.FromMinutes(15), "24", false, "DistractingChannel"));

            var controller = new VideoController(mockGuard, mockYouTube.Object);

            var result = await controller.CheckVideo("chan_block_123");

            var badRequestResult = Assert.IsType<BadRequestObjectResult>(result);
            var responseString = badRequestResult.Value?.ToString();
            
            Assert.Contains("Blocked", responseString);
            Assert.Contains("exceeded the limit for the 'DistractingChannel' channel", responseString);
        }
    }
}