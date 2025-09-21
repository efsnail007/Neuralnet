from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse

from .apps import BertClassifierConfig
from .models import Task


def index(request):
    tasks_list = Task.objects.all()
    context = {"tasks_list": tasks_list}
    return render(request, "bert_classifier/index.html", context)


def detail(request, task_id):
    task = get_object_or_404(Task, pk=task_id)
    latest_predictions = task.interactionml_set.order_by("-id")[:6][::-1]
    context = {"task": task, "latest_predictions": latest_predictions}
    return render(request, "bert_classifier/detail.html", context)


def predict(request, task_id):
    task = get_object_or_404(Task, pk=task_id)

    input_text = request.POST["text"]
    output = BertClassifierConfig.predictor.predict(input_text)

    task.interactionml_set.create(input_data=input_text, output_data=output)
    return HttpResponseRedirect(reverse("bert_classifier:detail", args=(task.id,)))
