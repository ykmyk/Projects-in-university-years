namespace FocusTube.Services
{
    /// <summary>
    /// Static helper class to translate YouTube's numeric IDs into more readable categories.
    /// </summary>
    public static class CategoryMapper
    {
        /// <summary>
        /// Maps a YouTube API category ID to an internal application category.
        /// Identifies if the category is inherently educational and it should bypass quotas.
        /// </summary>
        /// <param name="categoryId">The string representation of the numeric ID returned by Youtube API.</param>
        /// <returns>A tuple containing the mapped display Name and a boolean IsStudy flag.</returns>
        public static (string Name, bool IsStudy) GetCategoryInfo(string categoryId)
        {
            // We use a Switch Expression (Modern C#) to map IDs.
            // ID 27 = Education, 28 = Science/Tech. 
            // And we consider everything else is 'Noise' for this FocusTube application.
            return categoryId switch
            {
                // Core Study Categories
                "27" => ("Education", true),
                "28" => ("Science & Technology", true),

                // Entertainment & Distractions
                "1"  => ("Film & Animation", false),
                "2"  => ("Autos & Vehicles", false),
                "10" => ("Music", false),
                "15" => ("Pets & Animals", false),
                "17" => ("Sports", false),
                "18" => ("Short Movies", false),
                "19" => ("Travel & Events", false),
                "20" => ("Gaming", false),
                "21" => ("Videoblogging", false),
                "22" => ("People & Blogs", false),
                "23" => ("Comedy", false),
                "24" => ("Entertainment", false),
                "25" => ("News & Politics", false),
                "26" => ("Howto & Style", false),
                "29" => ("Nonprofits & Activism", false),
                
                // Movies, Shows & Shorts
                "30" => ("Movies", false),
                "31" => ("Anime/Animation", false),
                "32" => ("Action/Adventure", false),
                "33" => ("Classics", false),
                "34" => ("Comedy", false), // Note: YouTube API uses 34 for Comedy movies specifically
                "35" => ("Documentary", false),
                "36" => ("Drama", false),
                "37" => ("Family", false),
                "38" => ("Foreign", false),
                "39" => ("Horror", false),
                "40" => ("Sci-Fi/Fantasy", false),
                "41" => ("Thriller", false),
                "42" => ("Shorts", false),
                "43" => ("Shows", false),
                "44" => ("Trailers", false),

                // Fallback
                _    => ("General/Other", false) 
            };
        }
    }
}