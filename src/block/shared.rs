impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Clear<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Clear(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{Clear(self.0.valid())}
	type InnerModule=Clear<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Registry<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Registry(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{Registry(self.0.valid())}
	type InnerModule=Registry<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Shared<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Shared::map(&inner,|v|V::from_inner(v.clone()))}
	fn valid(&self)   ->Self::InnerModule       {Shared::map(self,V::valid)}
	type InnerModule=Shared<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Update<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Update(AutodiffModule::from_inner(inner.0))}
	fn valid(&self)->Self::InnerModule{Update(self.0.valid())}
	type InnerModule=Update<W>;
}
impl<V:Any+Default+Send> Default for Clear<V>{
	fn default()->Self{Self(Shared::default())}
}
impl<V:Any+Send> Default for Registry<V>{
	fn default()->Self{Self(Vec::new())}
}
impl<V:Any+Default+Send> Default for Shared<V>{
	fn default()->Self{Self::new(V::default())}
}
impl<V:Any+Default+Send> Default for Update<V>{
	fn default()->Self{Self(Shared::default())}
}
impl<'a,V:Any+DeserializeOwned+Send> Deserialize<'a> for Shared<V>{
	fn deserialize<D:Deserializer<'a>>(deserializer:D)->Result<Self,D::Error>{
		#[derive(Deserialize)]
		struct Serial<V>{inner:Option<Arc<Mutex<V>>>,generation:usize,lineage:u64}
		let serial:Serial<V>=Deserialize::deserialize(deserializer)?;

		let key=ShareKey{
			generation:serial.generation,lineage:serial.lineage,
			vtype:typeid::of::<V>()
		};
		let inner=serial.inner;
		let mut primary=false;

		let inner=inner.inspect(|v|{
			if let Some(maphandle)=SHARE_MAP.get(){
				maphandle.remove(&key.with_type::<ReserialFallback>());
			}
			primary=true;
			put_global_layer(key,v.clone());
		}).map(OnceCell::from).or_else(||{
			Some(get_global_layer(&key)?.into())
		}).or_else(||{
			let maphandle=SHARE_MAP.get()?;

			let fallbackhandle=maphandle.get(&key.with_type::<ReserialFallback>())?;
			let fallbackvalue=fallbackhandle.downcast_ref::<ReserialFallback>()?()?;
			let layervalue=rmp_serde::from_slice(&fallbackvalue).map(|x|Arc::new(Mutex::new(x))).ok();

			Some(layervalue.inspect(|v|put_global_layer(key,v.clone()))?.into())
		}).unwrap_or_else(OnceCell::new);

		Ok(Self{inner,key,primary})
	}
}
impl<B:Backend> BlockVariant<B> for Cache<B>{
	fn clear(&mut self){self.inner=None}
	fn detach_cache(&mut self){self.inner=self.inner.take().map(|x|x.detach())}
	fn forward(&self,input:Value<B>)->Value<B>{input}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		self.inner=Some(input.clone());
		input
	}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Cache<C>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Clear<V>{
	fn forward(&self,input:Value<B>)->Value<B>{
		self.0._edit(V::clear);
		input
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		self.0.edit(V::clear);
		input
	}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Clear<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Registry<V>{
	fn forward(&self,input:Value<B>)->Value<B>{input}
	fn supports(&self,_encoding:u64)->bool{true}
	type BlockWith<C:Backend>=Registry<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Shared<V>{
	fn clear(&mut self){self.visit_mut(|l|l.clear())}
	fn detach_cache(&mut self){self.visit_mut(|l|l.detach_cache())}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.get(|layer|layer.embed(input,inputclasses,inputencoding))}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.edit(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn encoding_hint(&self)->Option<u64>{self.get(|layer|layer.encoding_hint())}
	fn forward(&self,input:Value<B>)->Value<B>{self.get(|layer|layer.forward(input))}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.edit(|layer|layer.forward_mut(input))}
	fn supports(&self,encoding:u64)->bool{self.get(|layer|layer.supports(encoding))}
	type BlockWith<C:Backend>=Shared<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Update<V>{
	fn clear(&mut self){self.0.visit_mut(|l|l.clear())}
	fn detach_cache(&mut self){self.0.visit_mut(|l|l.detach_cache())}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.0._edit(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.0.edit(|layer|layer.embed_mut(input,inputclasses,inputencoding))}
	fn encoding_hint(&self)->Option<u64>{self.0.get(|layer|layer.encoding_hint())}
	fn forward(&self,input:Value<B>)->Value<B>{self.0._edit(|layer|layer.forward_mut(input))}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.0.edit(|layer|layer.forward_mut(input))}
	fn supports(&self,encoding:u64)->bool{self.0.get(|layer|layer.supports(encoding))}
	type BlockWith<C:Backend>=Update<V::BlockWith<C>>;
}
impl<V:Any+Clone+Send> Clone for Clear<V>{
	fn clone(&self)->Self{Self(self.0.clone())}
}
impl<V:Any+Clone+Send> Clone for Registry<V>{
	fn clone(&self)->Self{Self(self.0.clone())}
}
impl<V:Any+Clone+Send> Clone for Shared<V>{
	fn clone(&self)->Self{self.map(V::clone)}
}
impl<V:Any+Clone+Send> Clone for Update<V>{
	fn clone(&self)->Self{Self(self.0.clone())}
}
impl<B:Backend> From<Option<Value<B>>> for Cache<B>{
	fn from(inner:Option<Value<B>>)->Self{
		Self{inner}
	}
}
impl<B:Backend> From<Value<B>> for Cache<B>{
	fn from(inner:Value<B>)->Self{
		Self{inner:Some(inner)}
	}
}
impl<V:Any+Send> From<V> for Clear<V>{
	fn from(inner:V)->Self{Self(inner.into())}
}
impl<V:Any+Send> From<V> for Shared<V>{
	fn from(inner:V)->Self{Self::new(inner)}
}
impl<V:Any+Send> From<V> for Update<V>{
	fn from(inner:V)->Self{Self(inner.into())}
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Clear<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){Module::visit(&self.0,visitor)}
	type Record=<Option<V> as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Registry<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=<Vec<Shared<V>> as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Shared<V>{
	fn collect_devices(&self,mut devices:Vec<B::Device>)->Vec<B::Device>{
		self.visit(|v|devices=v.collect_devices(mem::take(&mut devices)));
		devices
	}
	fn fork(self,device:&B::Device)->Self{Shared::map(&self,|v|v.clone().fork(device))}
	fn into_record(self)->Self::Record{
		let mut layer=None;

		self.visit(|v|layer=Some(v.clone()));
		layer.into_record()
	}
	fn load_record(mut self,record:Self::Record)->Self{
		self.visit_mut(|v|{
			let layer:Option<V>=None;
			if let Some(l)=layer.load_record(record){*v=l}
		});
		self
	}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Shared::map(&self,|v|v.clone().map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Shared::map(&self,|v|v.clone().to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.visit(|v|v.visit(visitor))}
	type Record=<Option<V> as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Update<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){Module::visit(&self.0,visitor)}
	type Record=<Option<V> as Module<B>>::Record;
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Clear<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Registry<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Shared<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplay for Update<V>{}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Shared<V>{
	fn content(&self,content:Content)->Option<Content>{
		let mut c=None;
		self.visit(|l|c=l.content(content));

		c
	}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Clear<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Registry<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:Any+ModuleDisplay+Send> ModuleDisplayDefault for Update<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:Any+Send+Serialize> Serialize for Shared<V>{
	fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{
		#[derive(Serialize)]
		struct Serial<V:Serialize>{inner:Option<Arc<Mutex<V>>>,generation:usize,lineage:u64}

		let h=self.key;
		let fallback:Box<dyn Fn()->Option<Vec<u8>>+Send+Sync>=Box::new(move||{
			let h:Shared<V>=Shared{inner:OnceCell::new(),key:h,primary:false};
			rmp_serde::to_vec(&*h.get_inner_layer().lock().ok()?).ok()
		});

		SHARE_MAP.get_or_init(Default::default).insert(self.key.with_type::<ReserialFallback>(),Box::new(fallback));
		Serial{
			inner:self.inner.get().filter(|_|{
				if self.primary{
					dbg!(self.key);
				}
				self.primary
			}).cloned(),
			generation:self.key.generation,
			lineage:self.key.lineage
		}.serialize(serializer)
	}
}

impl<B:Backend> Cache<B>{
	/// get a clone of the inner value
	pub fn get_inner(&self)->Option<Value<B>>{self.inner.clone()}
	/// reference the inner value
	pub fn inner(&self)->&Option<Value<B>>{&self.inner}
	/// reference the inner value
	pub fn inner_mut(&mut self)->&mut Option<Value<B>>{&mut self.inner}
	/// convert into the inner value
	pub fn into_inner(self)->Option<Value<B>>{self.inner}
	/// create a new empty cache. use from(value) to create with an existing value inside
	pub fn new()->Self{
		Self{inner:None}
	}
}
impl<V:Any+Send> Clear<V>{
	/// create another non primary share with the same key
	pub fn share(&self)->Self{Self(self.0.share())}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary
	pub fn share_swap(&mut self)->Self{Self(self.0.share_swap())}
}
impl<V:Any+Send> Shared<V>{
	fn _edit<F:FnOnce(&mut V)->Y,Y>(&self,f:F)->Y{
		let layerhandle=self.get_inner_layer();
		let mut layerhandle=layerhandle.lock().unwrap();

		f(&mut layerhandle)
	}
	/// edit a value from the inner layer
	pub fn edit<F:FnOnce(&mut V)->Y,Y>(&mut self,f:F)->Y{self._edit(|v|f(v))}
	/// get the inner layer
	fn get_inner_layer(&self)->Arc<Mutex<V>>{self.inner.get_or_init(||get_global_layer(&self.key).expect("A primary share must be alive at this point")).clone()}
	/// get a value from the inner layer
	pub fn get<F:FnOnce(&V)->Y,Y>(&self,f:F)->Y{self._edit(|v|f(v))}
	/// map the inner value
	pub fn map<F:FnOnce(&V)->Y,Y:Any+Send>(&self,f:F)->Shared<Y>{
		let generation=self.key.generation+1;
		let lineage=self.key.lineage^RELINE.get();
		let vtype=typeid::of::<Y>();

		let key=ShareKey::from_inner(generation,lineage,vtype);
		let primary=self.primary;

		let inner=get_global_layer(&key).unwrap_or_else(||{
			let v=Arc::new(Mutex::new(self.get(f)));
			put_global_layer(key,v.clone());

			v
		}).into();
		Shared{inner,key,primary}
	}
	/// visit the inner value if this is a primary share
	pub fn visit<F:FnOnce(&V)>(&self,f:F){
		if self.primary{self._edit(|v|f(v))}
	}
	/// visit the inner value if this is a primary share
	pub fn visit_mut<F:FnOnce(&mut V)>(&mut self,f:F){
		if self.primary{self._edit(|v|f(v))}
	}

	/// thin wraps Shared and uses its built in interior mutability to call clear when forward pass methods are called
	pub fn and_clear(self)->Clear<V>{Clear(self)}
	/// thin wraps Shared and uses its built in interior mutability to call mutable versions of functions even when shared versions are called
	pub fn and_update(self)->Update<V>{Update(self)}
	/// creates a new shared cache. convert a share to a clear to clear the cache on forward, and update to update the cache on forward
	pub fn cache<B:Backend>()->Self where V:From<Cache<B>>{Self::new(Cache::new().into())}
	/// get the share key
	pub fn get_key(&self)->ShareKey{self.key}
	/// make this share a primary share of its key. For correct map/visit/serial behavior, exactly one primary share should be included with each model, so this shouldn't be used unless a reference to the layer needs outside for serialization or something. Primary share status is preserved when cloning
	pub fn into_primary(mut self)->Self{
		self.make_primary();
		self
	}
	/// check if this is a primary share
	pub fn is_primary(&self)->bool{self.primary}
	/// loads from a file. unless this is a primary share, only the key will actually be saved
	pub fn load<P:AsRef<Path>>(path:P)->IOResult<Self> where V:DeserializeOwned{
		let path=path.as_ref();

		let file=File::open(path)?;
		let reader=BufReader::new(file);

		match rmp_decode::from_read(reader){Err(e)=>Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(x)=>Ok(x)}
	}
	/// make this share a primary share of its key.
	pub fn make_primary(&mut self){self.primary=true}
	/// make this share a secondary share of its key.
	pub fn make_secondary(&mut self){self.primary=false}
	/// create a new share from the inner layer. The result will be a 'primary' Shared reference that delegates to the inner module for mapping and visiting purposes, and its shares will be secondary shares referencing the same layer with the same key. For module map/visit methods to work correctly, exactly one primary share should be present per key per model. Primary share status is preserved when cloning
	pub fn new(inner:V)->Self{
		let inner=Arc::new(Mutex::new(inner));
		let key=ShareKey::new::<V>();
		let primary=true;

		put_global_layer(key,inner.clone());
		Self{inner:inner.into(),key,primary}
	}
	/// saves to a file. unless this is a primary share, only the key will actually be saved
	pub fn save<P:AsRef<Path>>(&self,path:P)->IOResult<()> where V:Serialize{
		let file=File::create(path)?;
		let mut writer=BufWriter::new(file);

		match rmp_encode::write(&mut writer,self){Err(e)=>Err(IOError::new(IOErrorKind::Other,e.to_string())),Ok(x)=>Ok(x)}
	}
	/// create another non primary share with the same key
	pub fn share(&self)->Self{
		Shared{
			inner:self.inner.clone(),key:self.key.clone(),
			primary:false
		}
	}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary. This can be useful when creating a collection of shares to avoid having to explicitly swap or take the original primary into somewhere
	pub fn share_swap(&mut self)->Self{
		let mut s=self.share();

		mem::swap(&mut s,self);
		s
	}
}
impl<V:Any+Send> Update<V>{
	/// create another non primary share with the same key
	pub fn share(&self)->Self{Self(self.0.share())}
	/// create another share with the same key, then swap it with self before returning, effectively taking the primary status of self and putting it in the returned value, leaving self non primary
	pub fn share_swap(&mut self)->Self{Self(self.0.share_swap())}
}
impl<V> Drop for Shared<V>{
	fn drop(&mut self){
		if let Some(layerhandle)=mem::replace(&mut self.inner,OnceCell::new()).into_inner()&&Arc::strong_count(&layerhandle)==2{
			let maphandle=SHARE_MAP.get_or_init(Default::default);

			maphandle.remove(&self.key);
			maphandle.remove(&self.key.with_type::<ReserialFallback>());
		}
	}
}

impl ShareKey{
	/// create a share key from the inner data
	pub fn from_inner(generation:usize,lineage:u64,vtype:TypeId)->Self{
		Self{generation,lineage,vtype}
	}
	/// get the generation number
	pub fn get_generation(&self)->usize{self.generation}
	/// get the lineage id
	pub fn get_lineage(&self)->u64{self.lineage}
	/// gets a secondary share of the layer associated with this key if it exists
	pub fn get_shared<V:Any+Send>(&self)->Option<Shared<V>>{
		let inner=get_global_layer(self)?.into();
		Some(Shared{inner,key:*self,primary:false})
	}
	/// get the type id
	pub fn get_type(&self)->TypeId{self.vtype}
	/// create a new generation of a new lineage with the type
	pub fn new<V>()->Self{
		Self{
			generation:0,
			lineage:rand::random(),
			vtype:typeid::of::<V>()
		}
	}
	/// change type
	pub fn with_type<V>(mut self)->Self{
		self.vtype=typeid::of::<V>();
		self
	}
}

/// make a clone of a model such that its shared layers have the same share pattern but with new lineages independent of the original. This function relies on thread local behavior, and in the highly unusual case of a multithreaded clone impl only shares on the calling thread will actually break lineage.
pub fn break_lineage<V:Clone+Send>(module:&V)->V{
	let reset=RELINE.get();
	RELINE.set(rand::random());

	let result=module.clone();
	RELINE.set(reset);

	result
}
fn get_global_layer<V:Any+Send>(key:&ShareKey)->Option<Arc<Mutex<V>>>{SHARE_MAP.get()?.get(key)?.downcast_ref().cloned()}
fn put_global_layer<V:Any+Send>(key:ShareKey,layer:Arc<Mutex<V>>){
	SHARE_MAP.get_or_init(Default::default).insert(key,Box::new(layer));
}

#[derive(Debug,Deserialize,Module,Serialize)]
#[repr(transparent)]
#[serde(bound="")]
/// layer that caches tensor values for potential reuse. intended to be shared and wrapped with update or clear. Supports all encodings. Wrap in Adapt or Only to limit encoding support// TODO I don't think this is useful
pub struct Cache<B:Backend>{inner:Option<Value<B>>}
#[derive(Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize="V:Any+DeserializeOwned+Send",serialize="V:Any+Send+Serialize"))]
/// layer that clears the cache of shared layer through interior mutability. This just thin wraps Shared and uses its built in interior mutability to call clear_cache whenever a forward pass method is called. Supports all encodings. Wrap in Adapt or Only to limit encoding support
pub struct Clear<V>(pub Shared<V>);
#[derive(Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize="V:Any+DeserializeOwned+Send",serialize="V:Any+Send+Serialize"))]
/// a place to put primary shares in a model that lacks obvious blocks to keep them in. This is effectively an identity block with a payload
pub struct Registry<V>(pub Vec<Shared<V>>);
/// TODO get rid of this when we no longer need to_backend
type ReserialFallback=Box<dyn Fn()->Option<Vec<u8>>+Send+Sync>;
#[derive(Clone,Copy,Debug,Eq,Hash,PartialEq)]
/// the key type for identifying shares
pub struct ShareKey{generation:usize,lineage:u64,vtype:TypeId}
#[derive(Debug)]
/// wrap the inner layer to allow parameter and value sharing. Similar to an Arc<Mutex<V>>, but to prevent duplicate visits in module visiting operations like visit, serialize etc, it has a notion of order, where only shares with the primary order for a key participate. However, in module mapping operations that produce new layers, like clone and valid, only the first shares mapped participate
/// shares with the same key reference the same layer. The key consists of a generation, lineage, and type. Incrementing the generation on mapping operations allows whole models to be cloned while preserving internal share structure, rather than having the clone shares point at the originals. However, two clones or maps of the same model may still share layers with each other. Use break_lineage to clone them in a way that does not do this.
/// failing to fully deserialize the serialized layer will most likely result in panic on usage. Early dropping all primary shares of a key before deserialization of all secondary shares of the key will also result in panic on usage.
/// avoid reentrantly editing shares from within their own edit or forward operations as this will result in deadlock.
pub struct Shared<V>{inner:OnceCell<Arc<Mutex<V>>>,key:ShareKey,primary:bool}
#[derive(Debug,Deserialize,Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize="V:Any+DeserializeOwned+Send",serialize="V:Any+Send+Serialize"))]
/// layer that updates a shared layer through interior mutability. This just thin wraps Shared and uses its built in interior mutability to call mutable versions of functions even when shared versions are called. should generally pair with Clear to avoid breaking forward vs forward_mut assumptions
pub struct Update<V>(pub Shared<V>);

/// map share keys to share info
static SHARE_MAP:OnceLock<DashMap<ShareKey,Box<dyn Any+Send+Sync>>>=OnceLock::new();
thread_local!{
	static RELINE:Cell<u64>=const{Cell::new(0)};
}

use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	prelude::*,
	tensor::backend::AutodiffBackend
};
use dashmap::DashMap;
use rmp_serde::{decode as rmp_decode,encode as rmp_encode};
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::DeserializeOwned};
use super::{BlockVariant,Value};
use std::{
	any::{Any,TypeId},cell::{Cell,OnceCell},fs::File,io::{BufReader,BufWriter,Error as IOError,ErrorKind as IOErrorKind,Result as IOResult},mem,path::Path,sync::{Arc,Mutex,OnceLock}
};
