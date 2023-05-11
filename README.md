### VLP_web

demo.py can run standalone

To serve using the web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. You can learn more about the architecture [here](docs/server_arch.md).

Here are the commands to follow in your terminal:

#### Launch the controller
```bash
python controller.py
```
This controller manages the distributed workers.

#### Launch the model worker(s)
```bash
python model_worker.py --model-name blip2
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

#### Launch the Gradio web server
```bash
python server_demo.py
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI. You can open your browser and chat with a model now.
If the models do not show up, try to reboot the gradio web server.

#### TODO
- Use generate_iterator to handle 
