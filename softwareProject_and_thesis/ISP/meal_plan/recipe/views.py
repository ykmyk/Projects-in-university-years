from django.views import generic
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import get_object_or_404

from .models import RecipeString, RecipeBasic
from .forms import RecipeCreateForm

UNIT_CHOICE = ["g","kg","ml","l","tsp","tbsp","cup","piece"]

# preprocessing of the text data
# to make one step = one paragraph with the new line character between
# for later process to convert it to Cooklang
def normalize_steps(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    # drop empty row
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

class OnlyYouMixin(UserPassesTestMixin):
    raise_exception = True

    def test_func(self):
        # attain one recipe data from embedded URL key
        # if it can'T, show error 404
        recipe = get_object_or_404(RecipeBasic, pk=self.kwargs['pk'])

        # compare login user and recipe creation user
        # if it's different, follow raise_exception setting
        return self.request.user == recipe.created_by

class IndexView(generic.TemplateView):
    template_name = "index.html"


class RecipeListView(LoginRequiredMixin, generic.ListView):
    model = RecipeBasic
    template_name = 'recipe_list.html'
    context_object_name = "recipes"

    def get_queryset(self):
        """
        - Show public recipes + my own private recipes
        - Optional filtering by privacy (?visibility=public|private|all)
        - Optional search by name (?q=... )
        - Optional search by ingredients (?ing=potato&ing=carrot)
        """
        user = self.request.user
        qs = RecipeBasic.objects.filter(created_by=self.request.user).order_by('-created_at')

        visibility = (self.request.GET.get("visibility") or "").lower()
        if visibility == "public":
            qs = qs.filter(is_public=True)
        elif visibility == "private":
            qs = qs.filter(created_by=user, is_public=False)

        q = self.request.GET.get("q")
        if q:
            qs = qs.filter(name__icontains=q)

        ings = self.request.GET.getlist("ing")
        # naive ingredient filter: require each name to appear in any ingredient entry
        for ing_name in ings:
            ing_name = ing_name.strip()
            if not ing_name:
                continue
            # JSONField contains on Postgres works well; on SQLite keep expectations modest
            qs = qs.filter(ingredients__icontains=ing_name)

        return qs
    
class RecipeDetailView(LoginRequiredMixin, OnlyYouMixin, generic.DetailView):
    model = RecipeBasic
    template_name = 'recipe_detail.html'
    context_object_name = "object"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        cook = getattr(self.object, "cook", None)
        raw = cook.raw_text if cook else ""
        ctx["cook"] = cook
        ctx["steps_list"] = [ln for ln in raw.split("\n") if ln.strip()]
        return ctx


class RecipeCreateView(LoginRequiredMixin, generic.CreateView):
    model = RecipeBasic
    template_name = 'recipe_create.html'
    form_class = RecipeCreateForm
    success_url = reverse_lazy('recipe:recipe_list')

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["unit_choices"] = UNIT_CHOICE
        return ctx

    def form_valid(self, form):
        form.request = self.request 
        recipe = form.save(commit=False)
        recipe.created_by = self.request.user
        recipe.save()

        # preprocess the whole text content of cooking steps with normalize_step
        steps_text = normalize_steps(self.request.POST.get("step_text", ""))

        # create a new raw text data
        # with the cooking step content after preprocessed to the paragraph style
        RecipeString.objects.update_or_create(
            recipe = recipe,
            defaults={"raw_text": steps_text}
        )

        messages.success(self.request, 'Recipe has been created')
        return super().form_valid(form)
    
    def form_invalid(self, form):
        messages.error(self.request, 'failed to create new recipe')
        return super().form_invalid(form)
        
    
class RecipeUpdateView(LoginRequiredMixin, OnlyYouMixin, generic.UpdateView):
    model = RecipeBasic
    template_name = 'recipe_update.html'
    form_class = RecipeCreateForm

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        cook = getattr(self.object, "cook", None)
        raw = cook.raw_text if cook else ""
        ctx["steps_list"] = [ln for ln in raw.split("\n") if ln.strip()]
        ctx["unit_choices"] = UNIT_CHOICE
        return ctx

    def get_initial(self):
        initial = super().get_initial()
        cook = RecipeString.objects.filter(recipe=self.object).first()
        if cook:
            initial["steps_text"] = cook.raw_text
        return initial

    def get_success_url(self):
        return reverse_lazy('recipe:recipe_detail', kwargs={'pk': self.kwargs['pk']})
    
    def form_valid(self, form):
        form.request = self.request
        recipe = form.save()

        # preprocess the whole text content of cooking steps with normalize_step
        steps_text = normalize_steps(self.request.POST.get("step_text", ""))

        # create a new raw text data
        # with the cooking step content after preprocessed to the paragraph style
        RecipeString.objects.update_or_create(
            recipe = recipe,
            defaults={"raw_text": steps_text}
        )


        messages.success(self.request, 'Recipe has been updated')
        return super().form_valid(form)
    
    def form_invalid(self, form):
        messages.error(self.request, "failed to update the recipe")
        return super().form_invalid(form)
    

class RecipeDeleteView(LoginRequiredMixin, OnlyYouMixin, generic.DeleteView):
    model = RecipeBasic
    template_name = 'recipe_delete.html'
    success_url = reverse_lazy('recipe:recipe_list')

    def delete(self, request, *args, **kwargs):
        messages.success(self.request, "Recipe has been deleted")
        return super().delete(request, *args, **kwargs)