from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import get_object_or_404

class OnlyYouMixin(UserPassesTestMixin):
    raise_exception = True

    def test_func(self):
        # attain one recipe data from embedded URL key
        # if it can'T, show error 404
        recipe = get_object_or_404(RecipeBasic, pk=self.kwargs['pk'])

        # compare login user and recipe creation user
        # if it's different, follow raise_exception setting
        return self.request.user == recipe.created_by
    
