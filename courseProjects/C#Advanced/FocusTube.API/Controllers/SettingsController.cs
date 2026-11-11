using Microsoft.AspNetCore.Mvc;
using FocusTube.Services;
using System.Collections.Generic;
using System;

namespace FocusTube.API.Controllers
{
    /// <summary>
    /// API Controller that is responsible for retrieving and updating 
    /// the user's hierarchical time limits.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class SettingsController : ControllerBase
    {
        private readonly WatchGuardService _guard;

        /// <summary>
        /// Initializes a new instance of the SettingsController.
        /// </summary>
        /// <param name="guard"> 
        /// WatchGuardService guard will handle the domain logic for analytics.
        /// </param>
        public SettingsController(WatchGuardService guard) { _guard = guard; }

        /// <summary>
        /// The object to store the hours and minutes limit temprarily to transfer data 
        /// between the frontend and backend.
        /// </summary>
        public class TimeSetting { public int Hours { get; set; } public int Minutes { get; set; } }
        
        /// <summary>
        /// The object to store the data temporarily for receiving 
        /// complex nested settings from the frontend
        /// </summary>
        public class SettingsUpdate
        {
            public TimeSetting Total { get; set; } = new();
            public Dictionary<string, TimeSetting> Categories { get; set; } = new();
            public Dictionary<string, TimeSetting> Channels { get; set; } = new();
        }

        /// <summary>
        /// Retrieves the user's current settings to generate the front-end forms.
        /// </summary>
        /// <returns>A JSON object containing current hours and minutes for all tracked limits.</returns>
        [HttpGet("current")]
        public IActionResult GetCurrentSettings()
        {
            TimeSetting ToDto(TimeSpan ts) => new TimeSetting { Hours = ts.Hours, Minutes = ts.Minutes };

            var current = new SettingsUpdate
            {
                Total = ToDto(_guard.GetTotalRemaining()),
                Categories = _guard.GetCategoryLimits().ToDictionary(k => k.Key, v => ToDto(v.Value)),
                Channels = _guard.GetChannelLimits().ToDictionary(k => k.Key, v => ToDto(v.Value))
            };
            return Ok(current);
        }

        /// <summary>
        /// Overwrites/Updates the user's current limits based on JSON input from the frontend.
        /// </summary>
        /// <param name="request">JSON style content with the new limit values</param>
        /// <returns>
        /// An OK response if it is successful, 
        /// or a Bad Request if the limits exceed 24 hours.
        /// </returns>
        [HttpPost("update")]
        public IActionResult UpdateLimits([FromBody] SettingsUpdate request)
        {
            try
            {
                _guard.UpdateTotalLimit(new TimeSpan(request.Total.Hours, request.Total.Minutes, 0));

                foreach (var cat in request.Categories)
                    _guard.UpdateCategoryLimit(cat.Key, new TimeSpan(cat.Value.Hours, cat.Value.Minutes, 0));

                foreach (var chan in request.Channels)
                    _guard.UpdateChannelLimit(chan.Key, new TimeSpan(chan.Value.Hours, chan.Value.Minutes, 0));

                return Ok(new { message = "All settings saved successfully!" });
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message); 
            }
        }
    }
}