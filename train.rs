impl MetricsRenderer for DontRender{
	fn manual_close(&mut self){}
	fn register_metric(&mut self,_definition:MetricDefinition){}
}
impl MetricsRendererEvaluation for DontRender{
	fn update_test(&mut self,_name:EvaluationName,_state:MetricState){}
	fn render_test(&mut self,_item:EvaluationProgress,_types:Vec<ProgressType>){}
}
impl MetricsRendererTraining for DontRender{
	fn update_train(&mut self,_state:MetricState){}
	fn update_valid(&mut self,_state:MetricState){}
	fn render_train(&mut self,_item:TrainingProgress,_types:Vec<ProgressType>){}
	fn render_valid(&mut self,_item:TrainingProgress,_types:Vec<ProgressType>){}
}

impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B,BlockWith<A>=V>> AutodiffModule<A> for LabeledImageTrain<A,V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		let (artifact,seq,encoding,height,imageext,labelext,render,tokenizer,train,width,valid)=(inner.artifact.clone(),inner.seq.clone(),inner.encoding.clone(),inner.height.clone(),inner.imageext.clone(),inner.labelext.clone(),inner.render.clone(),inner.tokenizer.clone(),inner.train.clone(),inner.width.clone(),inner.valid.clone());
		let model=V::from_inner(inner.model);
		let phantom=PhantomData;

		LabeledImageTrain{artifact,seq,encoding,height,imageext,labelext,model,phantom,render,tokenizer,train,width,valid}
	}
	fn valid(&self)->Self::InnerModule{
		let (artifact,seq,encoding,height,imageext,labelext,render,tokenizer,train,width,valid)=(self.artifact.clone(),self.seq.clone(),self.encoding.clone(),self.height.clone(),self.imageext.clone(),self.labelext.clone(),self.render.clone(),self.tokenizer.clone(),self.train.clone(),self.width.clone(),self.valid.clone());
		let model=self.model.valid();
		let phantom=PhantomData;

		LabeledImageTrain{artifact,seq,encoding,height,imageext,labelext,model,phantom,render,tokenizer,train,width,valid}
	}
	type InnerModule=LabeledImageTrain<B,W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B,BlockWith<A>=V>> AutodiffModule<A> for TokenSeqTrain<A,V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		let (artifact,seq,encoding,maxwindow,render,tokenizer,train,valid)=(inner.artifact.clone(),inner.seq.clone(),inner.encoding.clone(),inner.maxwindow.clone(),inner.render.clone(),inner.tokenizer.clone(),inner.train.clone(),inner.valid.clone());
		let model=V::from_inner(inner.model);
		let phantom=PhantomData;

		TokenSeqTrain{artifact,seq,encoding,maxwindow,model,phantom,render,tokenizer,train,valid}
	}
	fn valid(&self)->Self::InnerModule{
		let (artifact,seq,encoding,maxwindow,render,tokenizer,train,valid)=(self.artifact.clone(),self.seq.clone(),self.encoding.clone(),self.maxwindow.clone(),self.render.clone(),self.tokenizer.clone(),self.train.clone(),self.valid.clone());
		let model=self.model.valid();
		let phantom=PhantomData;

		TokenSeqTrain{artifact,seq,encoding,maxwindow,model,phantom,render,tokenizer,train,valid}
	}
	type InnerModule=TokenSeqTrain<B,W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B,BlockWith<A>=V>> TrainStep for LabeledImageTrain<A,V>{
	fn step(&self,input:LabeledImageBatch<A>)->TrainOutput<ClassificationOutput<A>>{
		let (input,target)=(input.get_images(),input.get_labels());
		let output=self.model.forward(Value::new(input,self.encoding));
																				// get a device and reshape output and target so we have the loss inputs in the forms the library functions expect
		let additionalloss=output.get_loss();
		let device=target.device();
		let output=output.get_data();
		let target=target.reshape([-1]);
																				// compute the cross entropy loss
		let loss=CrossEntropyLoss::new(Some(0),&device);
		let loss=loss.forward(output.clone(),target.clone())/(output.dims()[output.rank()-1] as f32).ln();
																				// add additional loss gradients if necessary and wrap in a classification output
		let loss=if let Some(additionalloss)=additionalloss{
			let offset:f32=additionalloss.clone().into_scalar().elem();
			additionalloss+loss-offset
		}else{
			loss
		};
		let lossgrad=loss.clone().backward();
		let output=ClassificationOutput::new(loss,output,target);
																				// create a train output from the results
		TrainOutput::new(self,lossgrad,output)
	}
	type Input=LabeledImageBatch<A>;
	type Output=ClassificationOutput<A>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B,BlockWith<A>=V>> TrainStep for TokenSeqTrain<A,V>{
	fn step(&self,input:TokenSeqBatch<A>)->TrainOutput<ClassificationOutput<A>>{
		let (input,target)=(input.get_input(),input.get_target());
		let [_batch,seq]=input.dims();
		let window=seq.min(self.maxwindow);
																				// ensure even training on large amounts of seq despite our limitations by chunking into maxwindow and picking a chunk to descend on after inferring on all chunks up to it
		let (output,target)=if seq>window{
			let mut model=self.model.clone();
			input.split(seq,1).into_iter().zip(target.split(seq,1)).map(|(input,target)|{
				let output=model.embed_mut(input,self.tokenizer.len(),self.encoding);
				model.detach_cache();

				(output,target)
			}).nth(rand::random_range(0..window.div_ceil(seq))).unwrap()
		}else{
			(self.model.embed(input,self.tokenizer.len(),self.encoding),target)
		};
																				// get a device and reshape output and target so we have the loss inputs in the forms the library functions expect
		let additionalloss=output.get_loss();
		let device=target.device();
		let output=output.get_data();
		let target=target.reshape([-1]);
																				// compute the cross entropy loss
		let loss=CrossEntropyLoss::new(None,&device);
		let loss=loss.forward(output.clone(),target.clone())/(output.dims()[output.rank()-1] as f32).ln();
																				// add additional loss gradients if necessary and wrap in a classification output
		let loss=if let Some(additionalloss)=additionalloss{
			let offset:f32=additionalloss.clone().into_scalar().elem();
			additionalloss+loss-offset
		}else{
			loss
		};
		let lossgrad=loss.clone().backward();
		let output=ClassificationOutput::new(loss,output,target);
																				// create a train output from the results
		TrainOutput::new(self,lossgrad,output)
	}
	type Input=TokenSeqBatch<A>;
	type Output=ClassificationOutput<A>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B,BlockWith<A>=V>> TrainVariant<A> for LabeledImageTrain<A,V>{
	fn train(self,batch:usize,epochs:usize,learningrate:f32)->Self{
		let (artifact,seq,encoding,height,imageext,labelext,render,tokenizer,train,width,valid)=(self.artifact.clone(),self.seq.clone(),self.encoding.clone(),self.height.clone(),self.imageext.clone(),self.labelext.clone(),self.render.clone(),self.tokenizer.clone(),self.train.clone(),self.width.clone(),self.valid.clone());
		let batcher=DefaultBatcher;

		fs::create_dir_all(&artifact).ok();

		let model=self;
		let optimizer=AdamConfig::new().with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(0.1))).init();

		let (mut trainset,mut validset)=(LabeledImageData::new([1,height,width],seq),LabeledImageData::new([1,height,width],seq));

		for entry in WalkDir::new(&train).into_iter().filter_map(|e|e.ok().filter(|e|e.file_type().is_file())){
			let path=entry.path();
			if path.extension()!=Some((&*imageext).as_ref()){continue}

			trainset.open(entry.path(),path.with_extension(&*labelext),0,1,&tokenizer).unwrap()
		}
		for entry in WalkDir::new(&valid).into_iter().filter_map(|e|e.ok().filter(|e|e.file_type().is_file())){
			let path=entry.path();
			if path.extension()!=Some((&*imageext).as_ref()){continue}

			validset.open(entry.path(),path.with_extension(&*labelext),0,1,&tokenizer).unwrap()
		}

		let (trainloader,validloader)=(DataLoaderBuilder::new(batcher).batch_size(batch).shuffle(rand::random()),DataLoaderBuilder::new(batcher).batch_size(batch).shuffle(rand::random()));
		let (trainloader,validloader)=(trainloader.build(trainset),validloader.build(validset));
		let mut training=SupervisedTraining::new(&artifact,trainloader,validloader).metrics((AccuracyMetric::new(),LossMetric::new())).num_epochs(epochs);
		let scheduler=learningrate as f64;

		if !render{training=training.renderer(DontRender)}

		let model=training.launch(Learner::new(model,optimizer,scheduler)).model.model.to_backend::<A>();
		let phantom=PhantomData;

		LabeledImageTrain{artifact,seq,encoding,height,imageext,labelext,model,phantom,render,tokenizer,train,width,valid}
	}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B,BlockWith<A>=V>> TrainVariant<A> for TokenSeqTrain<A,V>{
	fn train(self,batch:usize,epochs:usize,learningrate:f32)->Self{
		let (artifact,seq,encoding,maxwindow,render,tokenizer,train,valid)=(self.artifact.clone(),self.seq.clone(),self.encoding.clone(),self.maxwindow.clone(),self.render.clone(),self.tokenizer.clone(),self.train.clone(),self.valid.clone());
		let batcher=DefaultBatcher;

		fs::create_dir_all(&artifact).ok();

		let model=self;
		let optimizer=AdamConfig::new().with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(0.1))).init();

		let (mut trainset,mut validset)=(TokenSeqData::new(seq),TokenSeqData::new(seq));

		for entry in WalkDir::new(&train).into_iter().filter_map(|e|e.ok().filter(|e|e.file_type().is_file())){
			trainset.open(entry.path(),0..u64::MAX,&tokenizer).unwrap()
		}
		for entry in WalkDir::new(&valid).into_iter().filter_map(|e|e.ok().filter(|e|e.file_type().is_file())){
			validset.open(entry.path(),0..u64::MAX,&tokenizer).unwrap()
		}
		//(trainset.open(&train,0..u64::MAX,&tokenizer).unwrap(),validset.open(&valid,0..u64::MAX,&tokenizer).unwrap());

		let (trainloader,validloader)=(DataLoaderBuilder::new(batcher).batch_size(batch).shuffle(rand::random()),DataLoaderBuilder::new(batcher).batch_size(batch).shuffle(rand::random()));
		let (trainloader,validloader)=(trainloader.build(trainset),validloader.build(validset));
		let mut training=SupervisedTraining::new(&artifact,trainloader,validloader).metrics((AccuracyMetric::new(),LossMetric::new())).num_epochs(epochs);
		let scheduler=learningrate as f64;

		if !render{training=training.renderer(DontRender)}

		let model=training.launch(Learner::new(model,optimizer,scheduler)).model.model.to_backend::<A>();
		let phantom=PhantomData;

		TokenSeqTrain{artifact,seq,encoding,maxwindow,model,phantom,render,tokenizer,train,valid}
	}
}
impl<B:Backend,V:BlockVariant<B>+ModuleDisplay> ModuleDisplay for LabeledImageTrain<B,V>{}
impl<B:Backend,V:BlockVariant<B>+ModuleDisplay> ModuleDisplay for TokenSeqTrain<B,V>{}
impl<B:Backend,V:BlockVariant<B>+ModuleDisplay> ModuleDisplayDefault for LabeledImageTrain<B,V>{
	fn content(&self,content:Content)->Option<Content>{self.model.content(content)}
}
impl<B:Backend,V:BlockVariant<B>+ModuleDisplay> ModuleDisplayDefault for TokenSeqTrain<B,V>{
	fn content(&self,content:Content)->Option<Content>{self.model.content(content)}
}
impl<B:Backend,V:BlockVariant<B>> Display for LabeledImageTrain<B,V>{
	fn fmt(&self,f:&mut Formatter<'_>)->Result<(),FmtError>{"LabeledImageTrain".fmt(f)}
}
impl<B:Backend,V:BlockVariant<B>> Display for TokenSeqTrain<B,V>{
	fn fmt(&self,f:&mut Formatter<'_>)->Result<(),FmtError>{"TokenSeqTrain".fmt(f)}
}
impl<B:Backend,V:BlockVariant<B>> From<V> for LabeledImageTrain<B,V>{
	fn from(model:V)->Self{
		let artifact=Arc::from(".artifact".as_ref());
		let seq=256;
		let encoding=model.encoding_hint().unwrap_or_default();
		let height=16;
		let imageext=Arc::from("png");
		let labelext=Arc::from("txt");
		let phantom=PhantomData;
		let render=true;
		let tokenizer=model.tokenizer_hint().unwrap_or_default();
		let train=Arc::from("train".as_ref());
		let width=16;
		let valid=Arc::from("valid".as_ref());

		Self{artifact,seq,encoding,height,imageext,labelext,model,phantom,render,tokenizer,train,width,valid}
	}
}
impl<B:Backend,V:BlockVariant<B>> From<V> for TokenSeqTrain<B,V>{
	fn from(model:V)->Self{
		let artifact=Arc::from(".artifact".as_ref());
		let seq=256;
		let encoding=model.encoding_hint().unwrap_or_default();
		let maxwindow=1024;
		let phantom=PhantomData;
		let render=true;
		let tokenizer=model.tokenizer_hint().unwrap_or_default();
		let train=Arc::from("train".as_ref());
		let valid=Arc::from("valid".as_ref());

		Self{artifact,seq,encoding,maxwindow,model,phantom,render,tokenizer,train,valid}
	}
}
impl<B:Backend,V:BlockVariant<B>> InferenceStep for LabeledImageTrain<B,V>{
	fn step(&self,input:LabeledImageBatch<B>)->ClassificationOutput<B>{
		let (input,target)=(input.get_images(),input.get_labels());
		let output=self.model.forward(Value::new(input,self.encoding));
																				// get a device and reshape output and target so we have the loss inputs in the forms the library functions expect
		let device=target.device();
		let output=output.get_data();
		let target=target.reshape([-1]);

		let loss=CrossEntropyLoss::new(Some(0),&device);
		let loss=loss.forward(output.clone(),target.clone())/(output.dims()[output.rank()-1] as f32).ln();
		let output=ClassificationOutput::new(loss,output,target);
																				// create a train output from the results
		output
	}
	type Input=LabeledImageBatch<B>;
	type Output=ClassificationOutput<B>;
}
impl<B:Backend,V:BlockVariant<B>> InferenceStep for TokenSeqTrain<B,V>{
	fn step(&self,input:TokenSeqBatch<B>)->ClassificationOutput<B>{
		let (input,target)=(input.get_input(),input.get_target());
		let [_batch,seq]=input.dims();
		let window=seq.min(self.maxwindow);
																				// ensure even training on different and large amounts of seq despite our limitations by chunking into maxwindow and picking a chunk to descend on after inferring on all chunks up to it
		let (output,target)=if seq>window{
			let mut model=self.model.clone();
			input.split(seq,1).into_iter().zip(target.split(seq,1)).map(|(input,target)|{
				let output=model.embed_mut(input,self.tokenizer.len(),self.encoding);
				model.detach_cache();

				(output,target)
			}).nth(rand::random_range(0..window.div_ceil(seq))).unwrap()
		}else{
			(self.model.embed(input,self.tokenizer.len(),self.encoding),target)
		};
																				// get a device and reshape output and target so we have the loss inputs in the forms the library functions expect
		let device=target.device();
		let output=output.get_data();
		let target=target.reshape([-1]);
																				// compute the cross entropy loss
		let loss=CrossEntropyLoss::new(None,&device);
		let loss=loss.forward(output.clone(),target.clone())/(output.dims()[output.rank()-1] as f32).ln();
		let output=ClassificationOutput::new(loss,output,target);
																				// create a train output from the results
		output
	}
	type Input=TokenSeqBatch<B>;
	type Output=ClassificationOutput<B>;
}
impl<B:Backend,V:BlockVariant<B>> LabeledImageTrain<B,V>{
	/// gets the sequence length
	pub fn get_seq(&self)->usize{self.seq}
	/// unwraps the inner model
	pub fn into_model(self)-><V as AutodiffModule<B>>::InnerModule where B:AutodiffBackend,V:AutodiffModule<B>{self.model.valid()}
	/// gets whether should render
	pub fn is_render(&self)->bool{self.render}
	/// moves the model to the Autodiff backend creates a new training structure
	pub fn new(model:V)->LabeledImageTrain<Autodiff<B>,V::BlockWith<Autodiff<B>>>{model.to_backend::<Autodiff<B>>().into()}
	/// sets whether the training should be renderered. a true value will lead to burn's tui. default=true
	pub fn set_render(&mut self,render:bool){self.render=render}
	/// sets the sequence length. labels will be padded or truncated. default=256
	pub fn set_seq(&mut self,seq:usize){self.seq=seq}
	/// sets the file to train from. default="train"
	pub fn set_train_path<P:AsRef<Path>>(&mut self,path:P){self.train=Arc::from(path.as_ref())}
	/// sets the file to validate from. default="valid"
	pub fn set_valid_path<P:AsRef<Path>>(&mut self,path:P){self.valid=Arc::from(path.as_ref())}
	/// references the train path
	pub fn train_path(&self)->&Path{&self.train}
	/// references the valid path
	pub fn valid_path(&self)->&Path{&self.valid}
	/// sets the render flag. default=true
	pub fn with_render(mut self,render:bool)->Self{
		self.set_render(render);
		self
	}
	/// sets the sequence length. labels will be padded or truncated. default=256
	pub fn with_seq(mut self,seq:usize)->Self{
		self.set_seq(seq);
		self
	}
	/// sets the file to train from. default="train"
	pub fn with_train_path<P:AsRef<Path>>(mut self,path:P)->Self{
		self.set_train_path(path);
		self
	}
	/// sets the file to validate from. default="valid"
	pub fn with_valid_path<P:AsRef<Path>>(mut self,path:P)->Self{
		self.set_valid_path(path);
		self
	}
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for LabeledImageTrain<B,V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.model.collect_devices(devices)}
	fn fork(mut self,device:&B::Device)->Self{
		self.model=self.model.fork(device);
		self
	}
	fn into_record(self)->Self::Record{self.model.into_record()}
	fn load_record(mut self,record:Self::Record)->Self{
		self.model=self.model.load_record(record);
		self
	}
	fn map<MM:ModuleMapper<B>>(mut self,mapper:&mut MM)->Self{
		self.model=self.model.map(mapper);
		self
	}
	fn to_device(mut self,device:&B::Device)->Self{
		self.model=self.model.to_device(device);
		self
	}
	fn visit<MM:ModuleVisitor<B>>(&self,visitor:&mut MM){self.model.visit(visitor)}
	type Record=V::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for TokenSeqTrain<B,V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.model.collect_devices(devices)}
	fn fork(mut self,device:&B::Device)->Self{
		self.model=self.model.fork(device);
		self
	}
	fn into_record(self)->Self::Record{self.model.into_record()}
	fn load_record(mut self,record:Self::Record)->Self{
		self.model=self.model.load_record(record);
		self
	}
	fn map<MM:ModuleMapper<B>>(mut self,mapper:&mut MM)->Self{
		self.model=self.model.map(mapper);
		self
	}
	fn to_device(mut self,device:&B::Device)->Self{
		self.model=self.model.to_device(device);
		self
	}
	fn visit<MM:ModuleVisitor<B>>(&self,visitor:&mut MM){self.model.visit(visitor)}
	type Record=V::Record;
}
impl<B:Backend,V:BlockVariant<B>> TokenSeqTrain<B,V>{
	/// gets the max window size. examples longer than max window size will divide the sequence into windows, choose a random window, read up to that window into the model, then do a train step on the chosen window
	pub fn get_max_window(&self)->usize{self.maxwindow}
	/// gets the sequence length
	pub fn get_seq(&self)->usize{self.seq}
	/// unwraps the inner model
	pub fn into_model(self)-><V as AutodiffModule<B>>::InnerModule where B:AutodiffBackend,V:AutodiffModule<B>{self.model.valid()}
	/// gets whether should render
	pub fn is_render(&self)->bool{self.render}
	/// moves the model to the Autodiff backend creates a new training structure
	pub fn new(model:V)->TokenSeqTrain<Autodiff<B>,V::BlockWith<Autodiff<B>>>{model.to_backend::<Autodiff<B>>().into()}
	/// sets the max window size. default=1024. examples longer than max window size will divide the sequence into windows, choose a random window, read up to that window into the model, then do a train step on the chosen window
	pub fn set_max_window(&mut self,maxwindow:usize){self.maxwindow=maxwindow}
	/// sets whether the training should be renderered. a true value will lead to burn's tui. default=true
	pub fn set_render(&mut self,render:bool){self.render=render}
	/// sets the sequence length. default=256
	pub fn set_seq(&mut self,seq:usize){self.seq=seq}
	/// sets the file to train from. default="train"
	pub fn set_train_path<P:AsRef<Path>>(&mut self,path:P){self.train=Arc::from(path.as_ref())}
	/// sets the file to validate from. default="valid"
	pub fn set_valid_path<P:AsRef<Path>>(&mut self,path:P){self.valid=Arc::from(path.as_ref())}
	/// references the train path
	pub fn train_path(&self)->&Path{&self.train}
	/// references the valid path
	pub fn valid_path(&self)->&Path{&self.valid}
	/// sets the max window size. default=1024. examples longer than max window size will divide the sequence into windows, choose a random window, read up to that window into the model, then do a train step on the chosen window
	pub fn with_max_window(mut self,maxwindow:usize)->Self{
		self.set_max_window(maxwindow);
		self
	}
	/// sets the render flag. default=true
	pub fn with_render(mut self,render:bool)->Self{
		self.set_render(render);
		self
	}
	/// sets the sequence length. default=256
	pub fn with_seq(mut self,seq:usize)->Self{
		self.set_seq(seq);
		self
	}
	/// sets the file to train from. default="train"
	pub fn with_train_path<P:AsRef<Path>>(mut self,path:P)->Self{
		self.set_train_path(path);
		self
	}
	/// sets the file to validate from. default="valid"
	pub fn with_valid_path<P:AsRef<Path>>(mut self,path:P)->Self{
		self.set_valid_path(path);
		self
	}
}

#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize)]
/// metric renderer for not rendering anything
pub struct DontRender;
#[derive(Clone,Debug,Deserialize,Serialize)]
#[serde(bound="")]
/// wrapper for training on labeled image sequence // TODO channels, pad tokens, make label to image possible in addition to image to label
pub struct LabeledImageTrain<B:Backend,V:BlockVariant<B>>{artifact:Arc<Path>,encoding:u64,height:usize,imageext:Arc<str>,labelext:Arc<str>,model:V,phantom:PhantomData<B>,render:bool,seq:usize,tokenizer:TokenDict,train:Arc<Path>,width:usize,valid:Arc<Path>}
#[derive(Clone,Debug,Deserialize,Serialize)]// TODO pad tokens
#[serde(bound="")]
/// wrapper for training on file token sequences
pub struct TokenSeqTrain<B:Backend,V:BlockVariant<B>>{artifact:Arc<Path>,encoding:u64,maxwindow:usize,model:V,phantom:PhantomData<B>,render:bool,seq:usize,tokenizer:TokenDict,train:Arc<Path>,valid:Arc<Path>}

pub trait TrainVariant<B:Backend>:'static+DeserializeOwned+Module<B>+Serialize{
	/// loads from a file
	fn load<P:AsRef<Path>>(path:P)->IOResult<Self>{
		let path=path.as_ref();

		let file=File::open(path)?;
		let reader=BufReader::new(file);

		match rmp_decode::from_read(reader){Err(e)=>Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(x)=>Ok(x)}
	}
	/// restores from a checkpoint file. this should be compatible with whatever format is used for checkpointing in self.train
	fn restore<P:AsRef<Path>>(self,path:P)->IOResult<Self>{
		let device=Default::default();
		match self.load_file(path.as_ref(),&CompactRecorder::new(),&device){
			Err(RecorderError::DeserializeError(e))=>Err(IOError::new(IOErrorKind::InvalidData,e)),
			Err(RecorderError::FileNotFound(e))=>Err(IOError::new(IOErrorKind::NotFound,e)),
			Err(RecorderError::Unknown(e))=>Err(IOError::new(IOErrorKind::Other,e)),
			Ok(m)=>Ok(m)
		}
	}
	/// saves to a file
	fn save<P:AsRef<Path>>(&self,path:P)->IOResult<()>{
		let file=File::create(path)?;
		let mut writer=BufWriter::new(file);

		match rmp_encode::write(&mut writer,self){Err(e)=>Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(x)=>Ok(x)}
	}
	/// trains the model. additional parameters may be included in self
	fn train(self,batch:usize,epochs:usize,learningrate:f32)->Self;
}

//pub use enumerate_model;
use burn::{
	backend::Autodiff,
	data::dataloader::DataLoaderBuilder,
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	nn::loss::CrossEntropyLoss,
	optim::AdamConfig,
	prelude::*,
	record::{CompactRecorder,RecorderError},
	tensor::backend::AutodiffBackend,
	train::{
		ClassificationOutput,InferenceStep,Learner,SupervisedTraining,TrainOutput,TrainStep,metric::{AccuracyMetric,LossMetric,MetricDefinition},renderer::{EvaluationName,EvaluationProgress,MetricState,MetricsRenderer,MetricsRendererEvaluation,MetricsRendererTraining,ProgressType,TrainingProgress}
	}
};
use crate::{
	block::{BlockVariant,Value},data::{DefaultBatcher,LabeledImageBatch,LabeledImageData,TokenSeqBatch,TokenSeqData}
};
use rmp_serde::{decode as rmp_decode,encode as rmp_encode};
use serde::{Deserialize,Serialize,de::DeserializeOwned};
use std::{
	fmt::{Display,Error as FmtError,Formatter},fs::{File,self},io::{BufReader,BufWriter,Error as IOError,ErrorKind as IOErrorKind,Result as IOResult},marker::PhantomData,path::Path,sync::Arc
};
use token_dict::TokenDict;
use walkdir::WalkDir;
