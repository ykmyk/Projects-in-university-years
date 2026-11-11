using FocusTube.Services;
using Xunit;

namespace FocusTube.Tests
{
    /// <summary>
    /// Contains unit tests for the CategoryMapper utility.
    /// Verifies if the CategoryMapper correctly parses the YouTube Category IDs 
    /// into internal business logic flags.
    /// </summary>
    public class CategoryTests
    {
        /// <summary>
        /// Uses an xUnit to test against the CategoryMapper.
        /// Ensures that study content(Education / Science & Technology) is allowed 
        /// and recreational content is correctly flagged.
        /// </summary>
        /// <param name="id">The numeric ID string returned by the YouTube API.</param>
        /// <param name="expectedName">The expected human-readable name of the category.</param>
        /// <param name="expectedStudy">The expected boolean indicating if the category bypasses quotas.</param>
        [Theory]
        [InlineData("27", "Education", true)]
        [InlineData("20", "Gaming", false)]
        public void Mapper_ShouldIdentifyStudyContentCorrectly(string id, string expectedName, bool expectedStudy)
        {
            var (name, isStudy) = CategoryMapper.GetCategoryInfo(id);
            
            Assert.Equal(expectedName, name);
            Assert.Equal(expectedStudy, isStudy);
        }
    }
}