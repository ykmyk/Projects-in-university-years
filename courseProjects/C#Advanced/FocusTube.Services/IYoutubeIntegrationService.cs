using System;
using System.Threading.Tasks;

namespace FocusTube.Services
{
    /// <summary>
    /// Interface defining the contract for YouTube data retrieval.
    /// This allows us to inject a real version in production and a "Mock" version during testing.
    /// </summary>
    public interface IYouTubeIntegrationService
    {
        Task<(TimeSpan Duration, string CategoryId, bool IsLive, string ChannelName)> GetVideoDetailsAsync(string videoId);
    }
}