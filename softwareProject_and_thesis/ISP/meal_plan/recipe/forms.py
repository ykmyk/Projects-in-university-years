import os

from django import forms

from .models import RecipeBasic

class RecipeCreateForm(forms.ModelForm):
    steps_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 6, "placeholder": "Step 1...\nStep 2...\n"}),
        label="How to cook (steps)"
    )

    class Meta:
        model = RecipeBasic
        fields = ["name", "servings", "duration_minutes", "is_public"]  # + you might have hours/min input instead
        labels = {
            "is_public": "publish this recipe", 
        }
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "servings": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
            "duration_minutes": forms.NumberInput(attrs={"class": "form-control", "min": 0}),
            "is_public": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }

    def save(self, commit=True):
        obj = super().save(commit=False)

        # Example: if you post arrays for ingredients:
        request = getattr(self, "request", None)
        if request:
            names = request.POST.getlist("ing-name[]")
            qtys  = request.POST.getlist("ing-qty[]")
            units = request.POST.getlist("ing-unit[]")
            ingredients = []
            for i, nm in enumerate(names):
                nm = (nm or "").strip()
                if not nm:
                    continue
                qraw = (qtys[i] or "").strip() if i < len(qtys) else ""
                unit = (units[i] or "").strip() if i < len(units) else ""
                quantity = float(qraw) if qraw else None
                ingredients.append({"order": i, "name": nm, "quantity": quantity, "unit": unit})
            obj.ingredients = ingredients

        if commit:
            obj.save()
        return obj