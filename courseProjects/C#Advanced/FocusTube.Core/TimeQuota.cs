using System;

namespace FocusTube.Core
{
    /// <summary>
    /// This TimeQuota class represents a secure, encapsulated domain model 
    /// for managing time-based quotas(the user time limit).
    /// Utilizes custom operator overloading to generate more intuitive mathematical interface 
    /// for the time limit computation such as time subtraction and limit threshold comparisons.
    /// </summary>
    public class TimeQuota
    {
        /// <summary>
        /// Gets the current remaining time. 
        /// This uses a private setter to ensure the time can only be modified via controlled operators,
        /// and protecting the domain state.
        /// </summary>
        public TimeSpan RemainingTime { get; private set; }

        /// <summary>
        /// Initializes a new instance of the TimeQuota class.
        /// </summary>
        /// <param name="remainingTime">
        /// The initial amounto of time allocated to the user for this specific quota.
        /// </param>
        public TimeQuota(TimeSpan remainingTime)
        {
            RemainingTime = remainingTime;
        }

        /// <summary>
        /// This custome operator overloads the subtraction (-) operator. 
        /// Allows the application to intuitively subtract a video's duration directly 
        /// from the quota object.
        /// </summary>
        /// <param name="quota">The current TimeQuota object.</param>
        /// <param name="duration">The TimeSpan (length of the requested video) to deduct.</param>
        /// <returns>A new TimeQuota instance containing the updated remaining time.</returns>
        public static TimeQuota operator -(TimeQuota quota, TimeSpan duration)
        {
            return new TimeQuota(quota.RemainingTime - duration);
        }

        /// <summary>
        /// This custome operator overloads the less-than (<) operator. 
        /// Provides a clean syntax to check if the remaining time in the quota is strictly less 
        /// than a requested video's duration to manage the time limit logic.
        /// </summary>
        /// <param name="left">The TimeQuota object to be evaluated.</param>
        /// <param name="right">The TimeSpan (length of the requested video) to compare against to "left".</param>
        /// <returns>
        /// True if the quota has less time remaining than the requested duration; 
        /// otherwise, false.
        /// </returns>
        public static bool operator <(TimeQuota left, TimeSpan right)
        {
            return left.RemainingTime < right;
        }

        /// <summary>
        /// This custome operator overloads the greater-than (>) operator. 
        /// Required by the C# compiler whenever the less-than (<) operator is overloaded 
        /// to ensure logical symmetry.
        /// </summary>
        /// <param name="left">The TimeQuota object to be evaluated.</param>
        /// <param name="right">The TimeSpan (length of the requested video) to compare against to "left".</param>
        /// <returns>
        /// True if the quota has strictly more time remaining than the requested duration; 
        /// otherwise, false.
        /// </returns>
        public static bool operator >(TimeQuota left, TimeSpan right)
        {
            return left.RemainingTime > right;
        }

        /// <summary>
        /// Overrides the default ToString method to provide a more readable representation 
        /// of the quota state.
        /// </summary>
        /// <returns>A formatted string displaying the remaining time.</returns>
        public override string ToString()
        {
            return $"Remaining Time: {RemainingTime}";
        }
    }
}