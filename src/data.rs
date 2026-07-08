impl<B:Backend,const N:usize> Deref for SerialFloatTensor<B,N>{
	fn deref(&self)->&Self::Target{&self.inner}
	type Target=Tensor<B,N>;
}
impl<B:Backend,const N:usize> DerefMut for SerialFloatTensor<B,N>{
	fn deref_mut(&mut self)->&mut Self::Target{&mut self.inner}
}
impl<B:Backend> Deref for SerialLinear<B>{
	fn deref(&self)->&Self::Target{&self.inner}
	type Target=Linear<B>;
}
impl<B:Backend> DerefMut for SerialLinear<B>{
	fn deref_mut(&mut self)->&mut Self::Target{&mut self.inner}
}
impl Dataset<Self> for ImageData{
	fn get(&self,n:usize)->Option<Self>{
		let size=self.channels*self.height*self.width;
		let start=n*size+self.range.start;
		let stop=start+size;

		if self.range.end<stop{return None}

		let mut result=self.clone();
		result.range=start..stop;

		Some(result)
	}
	fn len(&self)->usize{self.range.len()/(self.channels*self.height*self.width)}
}
impl Dataset<Self> for LabeledImageData{
	fn get(&self,n:usize)->Option<Self>{
		let imagedata=self.imagedata.get(n)?;
		let labeldata=self.labeldata.get(n).unwrap();

		Some(Self{imagedata,labeldata})
	}
	fn len(&self)->usize{self.imagedata.len()}
}
impl Dataset<Self> for TokenSeqData{
	fn get(&self,n:usize)->Option<Self>{
		let start=n*self.seq+self.range.start;
		let stop=start+self.seq+1;

		if self.range.end<stop{return None}

		let mut result=self.clone();
		result.range=start..stop;

		Some(result)
	}
	fn len(&self)->usize{self.range.len().saturating_sub(1)/self.seq}
}
impl ImageData{/// gets the data as mutable vec
	fn _data_mut<'a>(data:&'a mut Arc<Vec<f32>>,range:&mut Range<usize>)->&'a mut Vec<f32>{
		if (range.start,range.end)!=(0,data.len()){
			let datavec=data[range.clone()].to_vec();

			*range=0..datavec.len();
			*data=Arc::new(datavec);
		}
		Arc::make_mut(data)
	}
	/// references the data
	pub fn data(&self)->&[f32]{&self.data[self.range.clone()]}
	/// references the data
	pub fn data_mut(&mut self)->&mut [f32]{Self::_data_mut(&mut self.data,&mut self.range)}
	/// gets the dimensions of the images. [channels, height, width]
	pub fn get_dims(&self)->[usize;3]{[self.channels,self.height,self.width]}
	#[track_caller]
	/// creates a new empty ImageData. max channels is currently 4. height and width are limited to u32::MAX
	pub fn new(channels:usize,height:usize,width:usize)->Self{
		assert!(channels>0&&channels<=4);
		assert!(height>0&&height<=u32::MAX as usize);
		assert!(width>0&&width<=u32::MAX as usize);

		let data=Arc::new(Vec::new());
		let range=0..0;

		Self{channels,data,height,range,width}
	}
	#[track_caller]
	/// opens the image file and appends it to the data after 0pad/trunc and/or scaling to fit the dims. scale=0 for no scaling, scale=1 for fit scaling, scale=2 for fill scaling, When self has 1 channel the image is converted to greyscale, 2 channels -> grayscale with transparency alpha, 3 channels -> rgb, 4 channels -> rgba
	pub fn open<P:AsRef<Path>>(&mut self,path:P,scale:usize)->IOResult<()>{
		assert!(scale<3);

		let (channels,height,width)=(self.channels,self.height,self.width);
		let (cs,xs,ys)=(height*width,1,width);
		let mut image=match ImageReader::open(path)?.decode(){
			Err(e)=>if let ImageError::IoError(e)=e{
				return Err(e)
			}else{
				return Err(IOError::new(IOErrorKind::Other,e))
			},
			Ok(i)=>i
		};

		if height>u32::MAX as usize{panic!("ImageData is malformed: height  ={height}")}
		if width >u32::MAX as usize{panic!("ImageData is malformed: width   ={width }")}

		if scale==1{image=image.resize        (width as u32,height as u32,FilterType::Triangle)}
		if scale==2{image=image.resize_to_fill(width as u32,height as u32,FilterType::Triangle)}

		let (height,width)=(height.min(image.height() as usize),width.min(image.width() as usize));
		let data=Self::_data_mut(&mut self.data,&mut self.range);
		let start=data.len();
		let stop=channels*cs+start;

		data.resize(stop,0.0);
		self.range.end=stop;

		let data=&mut data[start..];
		match self.channels{
			1=>for (x,y, pixel) in GenericImageView::pixels(&image.into_luma8      ()).filter(|&(x,y,_p)|x<width as u32&&y<height as u32){
				let values=pixel.channels();
				data[0*cs+x as usize*xs+y as usize*ys]=values[0] as f32/255.0;
			}
			2=>for (x,y, pixel) in GenericImageView::pixels(&image.into_luma_alpha8()).filter(|&(x,y,_p)|x<width as u32&&y<height as u32){
				let values=pixel.channels();
				data[0*cs+x as usize*xs+y as usize*ys]=values[0] as f32/255.0;
				data[1*cs+x as usize*xs+y as usize*ys]=values[1] as f32/255.0;
			}
			3=>for (x,y, pixel) in GenericImageView::pixels(&image.into_rgb8       ()).filter(|&(x,y,_p)|x<width as u32&&y<height as u32){
				let values=pixel.channels();
				data[0*cs+x as usize*xs+y as usize*ys]=values[0] as f32/255.0;
				data[1*cs+x as usize*xs+y as usize*ys]=values[1] as f32/255.0;
				data[2*cs+x as usize*xs+y as usize*ys]=values[2] as f32/255.0;
			}
			4=>for (x,y, pixel) in GenericImageView::pixels(&image.into_rgba8      ()).filter(|&(x,y,_p)|x<width as u32&&y<height as u32){
				let values=pixel.channels();
				data[0*cs+x as usize*xs+y as usize*ys]=values[0] as f32/255.0;
				data[1*cs+x as usize*xs+y as usize*ys]=values[1] as f32/255.0;
				data[2*cs+x as usize*xs+y as usize*ys]=values[2] as f32/255.0;
				data[3*cs+x as usize*xs+y as usize*ys]=values[3] as f32/255.0;
			}
			c=>panic!("ImageData is malformed: channels={c}")
		}

		Ok(())
	}
}
impl LabeledImageData{
	/// references the data
	pub fn data(&self)->(&[f32],&[u32]){(self.imagedata.data(),self.labeldata.data())}
	/// gets the dimensions [channels, height, width]
	pub fn get_dims(&self)->[usize;3]{self.imagedata.get_dims()}
	/// gets the label sequence length
	pub fn get_seq(&self)->usize{self.labeldata.get_seq()}
	/// creates a new empty labeled image dataset
	pub fn new(dims:[usize;3],seq:usize)->Self{
		let [channels,height,width]=dims;

		Self{imagedata:ImageData::new(channels,height,width),labeldata:TokenSeqData::new(seq)}
	}
	/// opens a file and appends its tokens to the dataset
	pub fn open<P:AsRef<Path>,Q:AsRef<Path>>(&mut self,imagepath:P,labelpath:Q,pad:u32,scale:usize,tokenizer:&TokenDict)->IOResult<()>{
		let items=self.imagedata.len();

		self.imagedata.open(imagepath,scale)?;
		self.labeldata.open(labelpath,0..u64::MAX,tokenizer)?;
		self.labeldata.pad(pad);
		self.labeldata.truncate_to(items);

		Ok(())
	}
}
impl TokenSeqData{
	/// gets the data as mutable vec
	fn _data_mut<'a>(data:&'a mut Arc<FileVec<u32>>,range:&mut Range<usize>)->&'a mut FileVec<u32>{
		if (range.start,range.end)!=(0,data.len()){
			let mut datavec=FileVec::new();
			datavec.extend_from_slice(&data[range.clone()]);

			*range=0..datavec.len();
			*data=Arc::new(datavec);
		}
		Arc::make_mut(data)
	}
	#[track_caller]
	/// attemtps to create a new token dataset from the tokenized cache file. returns Err if the file doesn't exist or an error occurs
	pub fn cached<P:AsRef<Path>>(path:P,seq:usize)->IOResult<Self>{
		assert!(seq>0);
		if !fs::exists(&path)?{return Err(IOError::new(IOErrorKind::NotFound,"file doesn't exist"))}

		let data=Arc::new(unsafe{FileVec::open(path)?});
		let range=0..data.len();

		Ok(Self{data,range,seq})
	}
	/// references the data
	pub fn data(&self)->&[u32]{&self.data[self.range.clone()]}
	/// references the data
	pub fn data_mut(&mut self)->&mut [u32]{Self::_data_mut(&mut self.data,&mut self.range)}
	#[track_caller]
	/// creates an empty token dataset
	pub fn new(seq:usize)->Self{
		assert!(seq>0);

		let data=Arc::new(FileVec::new());
		let range=0..0;

		Self{data,range,seq}
	}
	/// opens a file and appends its tokens to the dataset. The file is read over the range of bytes, with the range offset so the end is not past the end of the file, and the length is truncated to the file length
	pub fn open<P:AsRef<Path>>(&mut self,path:P,mut range:Range<u64>,tokenizer:&TokenDict)->IOResult<()>{
		let data=Self::_data_mut(&mut self.data,&mut self.range);
		let mut file=File::open(path)?;

		let len=file.metadata()?.len();

		if len<range.end-range.start{range.start=range.end-len}
		if len<range.end{
			range.start-=range.end-len;
			range.end-=	 range.end-len;
		}

		let mut buffer=vec![0;(range.end-range.start).try_into().unwrap()];

		file.seek_relative(range.start as i64)?;
		file.read_exact(&mut buffer)?;

		if tokenizer.len()>256{data.extend(tokenizer.tokenize(buffer))}else{data.extend(buffer.into_iter().map(u32::from))};
		self.range.end=data.len();

		Ok(())
	}
	/// pads the data length to the next integer one more than a multiple of self.get_seq()
	pub fn pad(&mut self,val:u32){
		let len=self.data.len();
		let seq=self.seq;
											// find length to pad to, returning early if padding is unnecessary
		let paddedlen=match len%seq{
			0=>len+1,
			1=>return,
			n=>len-n+seq+1
		};
											// adjust the length
		Self::_data_mut(&mut self.data,&mut self.range).resize(paddedlen,val);
	}
	/// gets the sequence length. even if this contains only one sequence, it will still be less than self.data().len() due to the one extra target token at the end
	pub fn get_seq(&self)->usize{self.seq}
	/// sets whether the tokenized data should persist in a file
	pub fn set_persistent(&mut self,p:bool){Self::_data_mut(&mut self.data,&mut self.range).set_persistent(p)}
	/// truncates the data length to store n items. (len: n*self.get_seq()+1)
	pub fn truncate_to(&mut self,n:usize){
		let len=n*self.seq+1;

		if self.data.len()<=n{return}
		Self::_data_mut(&mut self.data,&mut self.range).truncate(len)
	}
}
impl<B:Backend> Batcher<B,ImageData,ImageBatch<B>> for DefaultBatcher{
	fn batch(&self,data:Vec<ImageData>,device:&B::Device)->ImageBatch<B>{ImageBatch::new(data,device)}
}
impl<B:Backend> Batcher<B,LabeledImageData,LabeledImageBatch<B>> for DefaultBatcher{
	fn batch(&self,data:Vec<LabeledImageData>,device:&B::Device)->LabeledImageBatch<B>{LabeledImageBatch::new(data,device)}
}
impl<B:Backend> Batcher<B,TokenSeqData,TokenSeqBatch<B>> for DefaultBatcher{
	fn batch(&self,data:Vec<TokenSeqData>,device:&B::Device)->TokenSeqBatch<B>{TokenSeqBatch::new(data,device)}
}
impl<B:Backend,const N:usize> From<SerialFloatTensor<B,N>> for Tensor<B,N>{
	fn from(tensor:SerialFloatTensor<B,N>)->Self{tensor.inner}
}
impl<B:Backend,const N:usize> From<Tensor<B,N>> for SerialFloatTensor<B,N>{
	fn from(inner:Tensor<B,N>)->Self{
		Self{inner}
	}
}
impl<B:Backend> From<Linear<B>> for SerialLinear<B>{
	fn from(inner:Linear<B>)->Self{
		Self{inner}
	}
}
impl<B:Backend> From<SerialLinear<B>> for Linear<B>{
	fn from(value:SerialLinear<B>)->Self{value.inner}
}
impl<B:Backend> From<Tensor<B,2,Int>> for TokenSeqBatch<B>{
	fn from(inner:Tensor<B,2,Int>)->Self{
		Self{data:inner}
	}
}
impl<B:Backend> ImageBatch<B>{
	/// gets the data tensor with dims [batch, chan, h, w]
	pub fn get_data(&self)->Tensor<B,4>{self.data.clone()}
	#[track_caller]
	/// creates a new image batch from the data
	pub fn new(data:Vec<ImageData>,device:&B::Device)->Self{
		assert!(data.len()>0);
		let [chan,h,w]=data[0].get_dims();
		data[1..].iter().for_each(|x|assert_eq!([chan,h,w],x.get_dims()));

		let data:Vec<f32>=data.iter().flat_map(|x|x.data().iter()).copied().collect();
		assert_eq!(data.len()%(chan*h*w),0);

		let batch=data.len()/(chan*h*w);
		Self{data:Tensor::from_data(TensorData::new(data,[batch,chan,h,w]),device)}
	}
}
impl<B:Backend> LabeledImageBatch<B>{
	#[track_caller]
	/// creates a new labeled image batch from the images and labels
	pub fn from_inner(images:Tensor<B,4>,labels:Tensor<B,2,Int>)->Self{
		assert_eq!(images.dims()[0],labels.dims()[0]);
		Self{images,labels}
	}
	/// gets the images
	pub fn get_images(&self)->Tensor<B,4>{self.images.clone()}
	/// gets the labels
	pub fn get_labels(&self)->Tensor<B,2,Int>{self.labels.clone()}
	#[track_caller]
	/// creates a labeled image batch from the data
	pub fn new(data:Vec<LabeledImageData>,device:&B::Device)->Self{
		let (imagedata,labeldata)=data.into_iter().map(|data|(data.imagedata,data.labeldata)).unzip();
		Self::from_inner(ImageBatch::new(imagedata,device).get_data(),TokenSeqBatch::new(labeldata,device).get_data())
	}
}
impl<B:Backend> TokenSeqBatch<B>{
	/// gets the sequence
	pub fn get_data(&self)->Tensor<B,2,Int>{self.data.clone()}
	/// gets the input portion of the sequence
	pub fn get_input(&self)->Tensor<B,2,Int>{self.data.clone().slice(s![..,..-1])}
	/// gets the target portion of the sequence
	pub fn get_target(&self)->Tensor<B,2,Int>{self.data.clone().slice(s![..,1..])}
	#[track_caller]
	/// creates a token sequence batch from the data
	pub fn new(data:Vec<TokenSeqData>,device:&B::Device)->Self{
		assert!(data.len()>0);
		let seq=data[0].get_seq();
		data[1..].iter().for_each(|x|assert_eq!(seq,x.get_seq()));

		let data:Vec<u32>=data.iter().flat_map(|x|x.data().iter()).copied().collect();
		assert_eq!(data.len()%(seq+1),0);

		let batch=data.len()/(seq+1);
		Self{data:Tensor::from_data(TensorData::new(data,[batch,seq+1]),device)}
	}
}
pub fn deserialize_conv2d<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<Conv2d<B>,D::Error>{
	let (bias,dilation,groups,kernel,padding,stride,weight):(Option<SerialFloatTensor<B,1>>,[usize;2],usize,[usize;2],PaddingConfig2d,[usize;2],SerialFloatTensor<B,4>)=Deserialize::deserialize(deserializer)?;
	let bias=bias.map(|x|Param::from_tensor(x.inner));
	let weight=Param::from_tensor(weight.inner);

	Ok(Conv2d{bias,dilation,groups,kernel_size:kernel,padding,stride,weight})
}
pub fn deserialize_embedding<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<Embedding<B>,D::Error>{
	let weight:SerialFloatTensor<B,2>=Deserialize::deserialize(deserializer)?;
	let weight=Param::from_tensor(weight.inner);

	Ok(Embedding{weight})
}
pub fn deserialize_layer_norm<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<LayerNorm<B>,D::Error>{
	let (bias,weight):(Option<SerialFloatTensor<B,1>>,SerialFloatTensor<B,1>)=Deserialize::deserialize(deserializer)?;
	let bias=bias.map(|t|Param::from_tensor(t.inner));
	let weight=Param::from_tensor(weight.inner);
	let mut norm=LayerNormConfig::new(10).init(&Default::default());

	(norm.beta,norm.gamma)=(bias,weight);
	Ok(norm)
}
pub fn deserialize_linear<'a,B:Backend,D:Deserializer<'a>>(deserializer:D)->Result<Linear<B>,D::Error>{
	let (bias,weight):(Option<SerialFloatTensor<B,1>>,SerialFloatTensor<B,2>)=Deserialize::deserialize(deserializer)?;
	let bias=bias.map(|t|Param::from_tensor(t.inner));
	let weight=Param::from_tensor(weight.inner);

	Ok(Linear{bias,weight})
}
pub fn deserialize_param<'a,B:Backend,D:Deserializer<'a>,const N:usize>(deserializer:D)->Result<Param<Tensor<B,N>>,D::Error>{
	let data:SerialFloatTensor<B,N>=Deserialize::deserialize(deserializer)?;
	Ok(Param::from_tensor(data.inner))
}
pub fn deserialize_max_pool2d<'a,D:Deserializer<'a>>(deserializer:D)->Result<MaxPool2d,D::Error>{
	let (ceil,dilation,kernel,padding,stride):(bool,[usize;2],[usize;2],PaddingConfig2d,[usize;2])=Deserialize::deserialize(deserializer)?;
	Ok(MaxPool2d{ceil_mode:ceil,dilation,kernel_size:kernel,padding,stride})
}
pub fn read_json_token_dict<P:AsRef<Path>>(path:P)->IOResult<TokenDict>{
	let file=File::open(path)?;
	let reader=BufReader::new(file);
	let vocab:Vec<String>=match rmp_decode::from_read(reader){Err(e)=>return Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(v)=>v};

	Ok(vocab.into_iter().collect())
}
pub fn serialize_conv2d<B:Backend,S:Serializer>(layer:&Conv2d<B>,serializer:S)->Result<S::Ok,S::Error>{
	let (dilation,groups,kernel,stride)=(layer.dilation,layer.groups,layer.kernel_size,layer.stride);
	let bias:Option<SerialFloatTensor<B,1>>=layer.bias.as_ref().map(|b|b.val().into());
	let padding:PaddingConfig2d=layer.padding.clone();
	let weight:SerialFloatTensor<B,4>=layer.weight.val().into();

	(bias,dilation,groups,kernel,padding,stride,weight).serialize(serializer)
}
pub fn serialize_embedding<B:Backend,S:Serializer>(layer:&Embedding<B>,serializer:S)->Result<S::Ok,S::Error>{
	let weight=SerialFloatTensor::from(layer.weight.val());
	weight.serialize(serializer)
}
pub fn serialize_layer_norm<B:Backend,S:Serializer>(layer:&LayerNorm<B>,serializer:S)->Result<S::Ok,S::Error>{(layer.beta.as_ref().map(|b|SerialFloatTensor::from(b.val())),SerialFloatTensor::from(layer.gamma.val())).serialize(serializer)}
pub fn serialize_linear<B:Backend,S:Serializer>(layer:&Linear<B>,serializer:S)->Result<S::Ok,S::Error>{
	let bias=layer.bias.as_ref().map(|b|SerialFloatTensor{inner:b.val()});
	let weight=SerialFloatTensor{inner:layer.weight.val()};

	(bias,weight).serialize(serializer)
}
pub fn serialize_max_pool2d<S:Serializer>(layer:&MaxPool2d,serializer:S)->Result<S::Ok,S::Error>{
	let (ceil,dilation,kernel,padding,stride)=(layer.ceil_mode,layer.dilation,layer.kernel_size,layer.padding.clone(),layer.stride);
	(ceil,dilation,kernel,padding,stride).serialize(serializer)
}
pub fn serialize_param<B:Backend,S:Serializer,const N:usize>(param:&Param<Tensor<B,N>>,serializer:S)->Result<S::Ok,S::Error>{
	let data=SerialFloatTensor::from(param.val());
	data.serialize(serializer)
}
#[derive(Clone,Copy,Debug,Default)]
/// batcher that just combines the data together into a tensor
pub struct DefaultBatcher;
#[derive(Debug,Module)]
/// batchified image data with dims [batch, chan, h, w]
pub struct ImageBatch<B:Backend>{data:Tensor<B,4>}
#[derive(Clone,Debug)]
/// data item or set of fixed size self targeted images // TODO ImageImageData for image targeted images
pub struct ImageData{channels:usize,data:Arc<Vec<f32>>,height:usize,range:Range<usize>,width:usize}
#[derive(Debug,Module)]
/// batch containing images and labels
pub struct LabeledImageBatch<B:Backend>{images:Tensor<B,4>,labels:Tensor<B,2,Int>}
#[derive(Clone,Debug)]
/// data item or set of labels and images
pub struct LabeledImageData{imagedata:ImageData,labeldata:TokenSeqData}
#[derive(Debug,Deserialize,Module,Serialize)]
#[repr(transparent)]
#[serde(bound="")]
/// wrapper to make tensors more conveniently serializable. They will be stored as tensor data and loaded with the default device
pub struct SerialFloatTensor<B:Backend,const N:usize>{
	#[serde(deserialize_with="deserialize_float_tensor")]
	#[serde(serialize_with="serialize_float_tensor")]
	pub inner:Tensor<B,N>
}
#[derive(Debug,Deserialize,Module,Serialize)]
#[repr(transparent)]
#[serde(bound="")]
pub struct SerialLinear<B:Backend>{
	#[serde(deserialize_with="deserialize_linear")]
	#[serde(serialize_with="serialize_linear")]
	pub inner:Linear<B>
}
#[derive(Debug,Module)]
/// batchified token seq data
pub struct TokenSeqBatch<B:Backend>{data:Tensor<B,2,Int>}
#[derive(Clone,Debug)]
/// data item or set of fixed length token sequences, where each token's target is the next token
pub struct TokenSeqData{data:Arc<FileVec<u32>>,range:Range<usize>,seq:usize}
pub use intertense::burn_ml::{deserialize_float_tensor,deserialize_int_tensor,serialize_float_tensor,serialize_int_tensor};
use burn::{
	data::{dataset::Dataset,dataloader::batcher::Batcher},
	module::Param,
	nn::{Embedding,LayerNormConfig,LayerNorm,Linear,PaddingConfig2d,modules::conv::Conv2d,modules::pool::MaxPool2d},
	prelude::*
};
use file_vec::FileVec;
use image::{GenericImageView,ImageReader,Pixel,error::ImageError,imageops::FilterType};
use serde::{Deserialize,Deserializer,Serialize,Serializer};
use rmp_serde::decode as rmp_decode;
use std::{
	fs::{File,self},io::{BufReader,Error as IOError,ErrorKind as IOErrorKind,Result as IOResult,Read,Seek},ops::{Deref,DerefMut,Range},path::Path,sync::Arc
};
use token_dict::TokenDict;
